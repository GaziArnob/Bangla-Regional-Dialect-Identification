# data_clean

Built by `../prepare_data.py` from the raw files in `../data/` (raw files untouched). All CSVs are UTF-8 with BOM, so Excel shows Bangla correctly.

## Files
| File | Rows | Dialects |
|---|---|---|
| all_sources.csv | 80,226 | all four sources stacked |
| banglaDial.csv | 60,719 | 12 labels (incl. Standard_Bangla) |
| vashantor.csv | 12,500 | Barisal, Chittagong, Mymensingh, Noakhali, Sylhet |
| onubad.csv | 2,940 | Barisal, Chittagong, Sylhet |
| locallang.csv | 4,067 | Barisal, Chittagong, Mymensingh, Rajshahi, Rangpur, Sylhet |

## Columns
`id, source, split, dialect, dialect_text, standard_text, english_text, text_norm, dup_in_source, in_other_sources`

- `split`: only Vashantor has one (train / validation / test), blank otherwise.
- `standard_text`: parallel Standard Bangla (Vashantor, ONUBAD, LocalLang; empty for BanglaDial).
- `english_text`: Vashantor and ONUBAD only.
- `text_norm`: dialect_text with punctuation removed, whitespace collapsed. Used for matching.
- `dup_in_source`: True if the same normalised text with the same dialect already appeared earlier in that source.
- `in_other_sources`: comma-separated list of other sources containing the same normalised text. BanglaDial overlaps almost all of the others.

## Cleaning applied
Unicode NFC, zero-width characters removed, whitespace collapsed, empty rows dropped. Duplicates are flagged, **not** removed.
The Rakhine column of the LocalLang file is left out (not a Bangla dialect).

## Not included
- Parallel Corpus for Dialect Bias Evaluation (download API failed; it is a merge of the sources above)
- BD-Dialect, Speech dataset (not downloaded)
