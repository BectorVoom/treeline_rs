//! Hand-rolled UBJSON type-tag decoder → `serde_json::Value` (XGB-02 / D-03).
//!
//! XGBoost's UBJSON format is decoded by this zero-dependency recursive-descent
//! decoder into a `serde_json::Value`, which the loader (`load_xgboost_ubjson`
//! in `lib.rs`) then feeds through `serde_json::from_value::<XgbModelJson>` — the
//! SAME structs and the SAME `de_f32` sentinel adapter the JSON path uses
//! (`json.rs`). Converging at `Value` (not a second struct path) is what makes
//! the UBJSON and JSON loads produce the IDENTICAL `Model` (D-01).
//!
//! ## Tag subset (RESEARCH §UBJSON Type-Tag Map — 14 tags)
//!
//! `Z`→Null; `T`/`F`→Bool; `i`(int8)/`U`(uint8)/`I`(int16)/`l`(int32)/`L`(int64)
//! →Number; `d`(float32)/`D`(float64)→Number; `C`(char)→1-char String;
//! `S`(length-tag + UTF-8)→String; `[`…`]`→Array; `{`…`}`→Object; `N`→no-op skip.
//! All multi-byte integers and floats are **big-endian** (UBJSON mandates
//! network byte order), decoded via `from_be_bytes`.
//!
//! ## Strongly-typed optimized containers (`$`/`#` fast path — Pitfall 4)
//!
//! XGBoost emits `[$<type>#<count>` everywhere: "all `count` elements share
//! `<type>`, per-element tags omitted." (e.g. `[$d#l<count>` is a float32 array,
//! how `split_conditions` is stored.) A naive per-element-tag decoder mis-parses
//! these. We peek for `$`/`#` after each container opener and take the fast path.
//! The `#`count is itself a tag-prefixed integer.
//!
//! ## Non-finite floats (Pitfall 5)
//!
//! UBJSON stores floats as raw IEEE-754, so a NaN/Inf arrives as an actual
//! `f32::NAN`/`INFINITY` BEFORE it becomes a `Value`. `serde_json::Value::Number`
//! cannot hold a non-finite f64 (it collapses to `Null`), which would silently
//! lose the value. So a non-finite `d`/`D` emits the sentinel `Value::String`
//! (`"@NaN@"`/`"@Inf@"`/`"@-Inf@"`) — the SAME sentinels the JSON pre-lexer
//! emits — routing through the shared `de_f32` adapter for numeric parity
//! (criterion-2).
//!
//! ## Safety (T-03-U01 / T-03-U02)
//!
//! Every byte read goes through `Cursor::take`, which returns `Err` on
//! truncation rather than indexing out of bounds. Every `$`/`#` container count
//! is validated against the bytes remaining BEFORE `Vec::with_capacity`, so a
//! hostile oversized count is rejected as a typed error, never an OOM
//! pre-allocation.

use serde_json::{Map, Number, Value};

use crate::error::XgbError;

/// A fallible byte cursor over the UBJSON stream. Every read is bounds-checked
/// and surfaces an `XgbError::Ubjson` on truncation (never an OOB panic).
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, pos: 0 }
    }

    /// Bytes remaining unread in the stream.
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn err(&self, detail: impl Into<String>) -> XgbError {
        XgbError::Ubjson {
            pos: self.pos,
            detail: detail.into(),
        }
    }

    /// Read exactly `n` bytes, advancing the cursor. Truncation → typed error.
    fn take(&mut self, n: usize) -> Result<&'a [u8], XgbError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| self.err("length overflow"))?;
        let slice = self.bytes.get(self.pos..end).ok_or_else(|| {
            self.err(format!(
                "truncated: need {n} bytes, {} remain",
                self.remaining()
            ))
        })?;
        self.pos = end;
        Ok(slice)
    }

    /// Read a single tag/marker byte.
    fn take_u8(&mut self) -> Result<u8, XgbError> {
        Ok(self.take(1)?[0])
    }

    /// Peek the next byte without advancing (None at end of stream).
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }
}

/// Decode a complete UBJSON byte stream into a `serde_json::Value` (D-03).
///
/// Returns `XgbError::Ubjson` on an unknown tag, a truncated read, or a `$`/`#`
/// container count that exceeds the remaining stream length. Skips leading `N`
/// no-op markers (the byte D-09 keys on for UBJSON detection).
pub fn decode_ubjson(bytes: &[u8]) -> Result<Value, XgbError> {
    let mut c = Cursor::new(bytes);
    decode_value(&mut c)
}

/// Decode one value at the cursor (reads its leading type tag).
fn decode_value(c: &mut Cursor) -> Result<Value, XgbError> {
    let tag = c.take_u8()?;
    decode_with_tag(c, tag)
}

/// Decode one value given an already-consumed type `tag`. Shared by the
/// per-element path and the `$`-typed-container fast path (which reads the type
/// tag once, up front).
fn decode_with_tag(c: &mut Cursor, tag: u8) -> Result<Value, XgbError> {
    match tag {
        b'Z' => Ok(Value::Null),
        b'T' => Ok(Value::Bool(true)),
        b'F' => Ok(Value::Bool(false)),
        b'N' => {
            // No-op: skip and decode the next real value.
            decode_value(c)
        }
        // Integer tags → Value::Number (i64/u64).
        b'i' => {
            let v = i8::from_le_bytes([c.take_u8()?]);
            Ok(Value::Number(Number::from(v as i64)))
        }
        b'U' => {
            let v = c.take_u8()?;
            Ok(Value::Number(Number::from(v as u64)))
        }
        b'I' => {
            let b = c.take(2)?;
            let v = i16::from_be_bytes([b[0], b[1]]);
            Ok(Value::Number(Number::from(v as i64)))
        }
        b'l' => {
            let b = c.take(4)?;
            let v = i32::from_be_bytes([b[0], b[1], b[2], b[3]]);
            Ok(Value::Number(Number::from(v as i64)))
        }
        b'L' => {
            let b = c.take(8)?;
            let v = i64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
            Ok(Value::Number(Number::from(v)))
        }
        // Float tags → Value::Number, OR the non-finite sentinel String (Pitfall 5).
        b'd' => {
            let b = c.take(4)?;
            let v = f32::from_be_bytes([b[0], b[1], b[2], b[3]]);
            Ok(float_to_value(v as f64))
        }
        b'D' => {
            let b = c.take(8)?;
            let v = f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
            Ok(float_to_value(v))
        }
        b'C' => {
            // Char: a single byte → 1-char String.
            let v = c.take_u8()?;
            Ok(Value::String((v as char).to_string()))
        }
        b'S' => Ok(Value::String(decode_string(c)?)),
        b'[' => decode_array(c),
        b'{' => decode_object(c),
        other => Err(c.err(format!(
            "unknown UBJSON type tag {:#x} ({:?})",
            other, other as char
        ))),
    }
}

/// Map a decoded float to a `Value`, emitting the shared NaN/Inf sentinel
/// STRING for non-finite values so it lands on the same `de_f32` adapter as the
/// JSON path (Pitfall 5 / criterion-2).
fn float_to_value(v: f64) -> Value {
    if v.is_finite() {
        // serde_json::Number::from_f64 returns None only for non-finite — finite
        // here, so the unwrap is safe.
        Number::from_f64(v)
            .map(Value::Number)
            .unwrap_or(Value::Null)
    } else if v.is_nan() {
        Value::String("@NaN@".to_string())
    } else if v.is_sign_positive() {
        Value::String("@Inf@".to_string())
    } else {
        Value::String("@-Inf@".to_string())
    }
}

/// Read a tag-prefixed length/count integer (`i`/`U`/`I`/`l`/`L`), as used by
/// `S` string lengths and `#` container counts. Negative lengths are rejected.
fn decode_length(c: &mut Cursor) -> Result<usize, XgbError> {
    let tag = c.take_u8()?;
    let v: i64 = match tag {
        b'i' => i8::from_le_bytes([c.take_u8()?]) as i64,
        b'U' => c.take_u8()? as i64,
        b'I' => {
            let b = c.take(2)?;
            i16::from_be_bytes([b[0], b[1]]) as i64
        }
        b'l' => {
            let b = c.take(4)?;
            i32::from_be_bytes([b[0], b[1], b[2], b[3]]) as i64
        }
        b'L' => {
            let b = c.take(8)?;
            i64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        }
        other => {
            return Err(c.err(format!(
                "expected an integer length/count tag, got {:#x} ({:?})",
                other, other as char
            )));
        }
    };
    usize::try_from(v).map_err(|_| c.err(format!("negative length/count {v}")))
}

/// Decode an `S` string body (length-tag + UTF-8 bytes).
fn decode_string(c: &mut Cursor) -> Result<String, XgbError> {
    let len = decode_length(c)?;
    let bytes = c.take(len)?;
    String::from_utf8(bytes.to_vec()).map_err(|e| c.err(format!("invalid UTF-8 in string: {e}")))
}

/// Validate a declared container `count` against the bytes that could possibly
/// remain, BEFORE pre-allocating (T-03-U01 DoS mitigation). Even the smallest
/// element is at least 1 byte, so a count exceeding `remaining` is impossible
/// and is rejected as a typed error rather than a giant `Vec::with_capacity`.
fn checked_capacity(c: &Cursor, count: usize) -> Result<usize, XgbError> {
    if count > c.remaining() {
        return Err(c.err(format!(
            "declared container count {count} exceeds {} remaining bytes",
            c.remaining()
        )));
    }
    Ok(count)
}

/// Decode an array body (the `[` opener already consumed), handling the
/// `$`type / `#`count optimized form (Pitfall 4) and the unoptimized
/// per-element-tag form (terminated by `]`).
fn decode_array(c: &mut Cursor) -> Result<Value, XgbError> {
    // Optimized: `$<type>` (optional) then `#<count>`.
    let elem_type = if c.peek() == Some(b'$') {
        c.take_u8()?; // consume '$'
        Some(c.take_u8()?) // the element type tag
    } else {
        None
    };

    if c.peek() == Some(b'#') {
        c.take_u8()?; // consume '#'
        let count = decode_length(c)?;
        let cap = checked_capacity(c, count)?;
        let mut arr = Vec::with_capacity(cap);
        if let Some(t) = elem_type {
            // All elements share `t`; per-element tags omitted.
            for _ in 0..count {
                arr.push(decode_with_tag(c, t)?);
            }
        } else {
            // `#`count without `$`type: each element still carries its own tag.
            for _ in 0..count {
                arr.push(decode_value(c)?);
            }
        }
        return Ok(Value::Array(arr));
    }

    if elem_type.is_some() {
        // `$` without `#` is malformed per the UBJSON optimized-container spec.
        return Err(c.err("typed array '$' marker without a '#' count"));
    }

    // Unoptimized: read tagged values until the `]` terminator.
    let mut arr = Vec::new();
    loop {
        match c.peek() {
            Some(b']') => {
                c.take_u8()?;
                break;
            }
            None => return Err(c.err("unterminated array (missing ']')")),
            Some(_) => arr.push(decode_value(c)?),
        }
    }
    Ok(Value::Array(arr))
}

/// Decode an object body (the `{` opener already consumed), handling the
/// `$`type / `#`count optimized form and the unoptimized form (terminated by
/// `}`). Object keys are bare UBJSON strings (length-tag + UTF-8, NO `S` tag).
fn decode_object(c: &mut Cursor) -> Result<Value, XgbError> {
    let elem_type = if c.peek() == Some(b'$') {
        c.take_u8()?;
        Some(c.take_u8()?)
    } else {
        None
    };

    if c.peek() == Some(b'#') {
        c.take_u8()?;
        let count = decode_length(c)?;
        // Each member is at least key-length(1) + key(0) + value(1) bytes; the
        // raw count guard is a conservative lower bound against OOM.
        checked_capacity(c, count)?;
        let mut map = Map::new();
        for _ in 0..count {
            let key = decode_string(c)?; // object key: bare length-prefixed string
            let val = match elem_type {
                Some(t) => decode_with_tag(c, t)?,
                None => decode_value(c)?,
            };
            map.insert(key, val);
        }
        return Ok(Value::Object(map));
    }

    if elem_type.is_some() {
        return Err(c.err("typed object '$' marker without a '#' count"));
    }

    let mut map = Map::new();
    loop {
        match c.peek() {
            Some(b'}') => {
                c.take_u8()?;
                break;
            }
            None => return Err(c.err("unterminated object (missing '}')")),
            Some(_) => {
                let key = decode_string(c)?;
                let val = decode_value(c)?;
                map.insert(key, val);
            }
        }
    }
    Ok(Value::Object(map))
}
