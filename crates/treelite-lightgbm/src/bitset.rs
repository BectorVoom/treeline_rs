//! `BitsetToList` — the LightGBM categorical-split bitset decoder (LGB-02).
//!
//! Ports `BitsetToList` VERBATIM from
//! `treelite-mainline/src/model_loader/lightgbm.cc:210-221`:
//!
//! ```cpp
//! inline std::vector<std::uint32_t> BitsetToList(std::uint32_t const* bits, std::size_t nslots) {
//!   std::vector<std::uint32_t> result;
//!   std::size_t const nbits = nslots * 32;
//!   for (std::size_t i = 0; i < nbits; ++i) {
//!     std::size_t const i1 = i / 32;
//!     std::uint32_t const i2 = static_cast<std::uint32_t>(i % 32);
//!     if ((bits[i1] >> i2) & 1) {
//!       result.push_back(static_cast<std::uint32_t>(i));
//!     }
//!   }
//!   return result;
//! }
//! ```
//!
//! The word/bit order is load-bearing: bit `i` lives in word `bits[i / 32]` at
//! bit position `i % 32` (LSB-first within each word). Any deviation silently
//! mis-decodes categories. This decoder is DELIBERATELY NOT shared with the
//! HistGradientBoosting `check(bitmap, ...)` sibling — that loader uses a
//! DIFFERENT bit layout (04-PATTERNS No-Analog-Found).
//!
//! **Bounds safety (T-04-11):** upstream walks `i` over `nslots * 32` bits and
//! indexes `bits[i / 32]` for `i / 32 ∈ [0, nslots)`. The caller slices exactly
//! `nslots` words out of `cat_threshold`, so `nslots == bits.len()` and the
//! index never escapes the slice. We take `bits: &[u32]` and derive `nslots`
//! from `bits.len()`, so the index is structurally in-bounds — there is no path
//! that reads past the bitset word array.

/// Decode a categorical-split bitset word array into its exact category list.
///
/// `bits` is the slice of `nslots = bits.len()` `u32` words for one categorical
/// split (the caller slices `cat_threshold[cat_boundaries[k]..cat_boundaries[k+1]]`).
/// Returns every category `i ∈ [0, bits.len() * 32)` whose bit is set, in
/// ascending order — exactly the upstream `BitsetToList(bits.data(), nslots)`.
pub fn bitset_to_list(bits: &[u32]) -> Vec<u32> {
    let nslots = bits.len();
    let nbits = nslots * 32;
    let mut result: Vec<u32> = Vec::new();
    for i in 0..nbits {
        let i1 = i / 32; // word index (always < nslots, structurally in-bounds).
        let i2 = (i % 32) as u32; // bit position within the word (LSB-first).
        // `bits[i1]` is in-bounds because i1 < nbits/32 == nslots == bits.len().
        if (bits[i1] >> i2) & 1 == 1 {
            result.push(i as u32);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bitset_yields_no_categories() {
        assert_eq!(bitset_to_list(&[]), Vec::<u32>::new());
    }

    #[test]
    fn single_word_lsb_first_order() {
        // word = 0b1011 → bits 0, 1, 3 set (LSB-first).
        assert_eq!(bitset_to_list(&[0b1011]), vec![0, 1, 3]);
    }

    #[test]
    fn high_bit_of_first_word() {
        // bit 31 (the MSB of word 0) set → category 31.
        assert_eq!(bitset_to_list(&[1u32 << 31]), vec![31]);
    }

    #[test]
    fn second_word_offsets_by_32() {
        // word 0 = 0 (no categories 0..31); word 1 bit 0 set → category 32.
        assert_eq!(bitset_to_list(&[0, 0b1]), vec![32]);
        // word 1 bit 5 set → category 32 + 5 = 37.
        assert_eq!(bitset_to_list(&[0, 1u32 << 5]), vec![37]);
    }

    #[test]
    fn reference_two_word_bitset_exact() {
        // A reference bitset: word 0 has bits {2, 4}, word 1 has bits {0, 1}.
        // Categories: 2, 4 (word 0) and 32, 33 (word 1).
        let bits = [(1u32 << 2) | (1u32 << 4), 0b11];
        assert_eq!(bitset_to_list(&bits), vec![2, 4, 32, 33]);
    }

    #[test]
    fn all_bits_set_single_word() {
        let got = bitset_to_list(&[u32::MAX]);
        let expected: Vec<u32> = (0..32).collect();
        assert_eq!(got, expected);
    }
}
