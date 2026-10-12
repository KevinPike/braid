# rune-seal

Rune codes use the 28-character alphabet `0123456789ACDEFHJKLMNPRTVWXY`. A character's value is its position in
that string, counting from 0 (`0` is 0, `9` is 9, `A` is 10, `C` is 11, `Y` is 27).

`seal(code: str) -> str` returns the code's normal form, then `-`, then its check character. Rules:

1. Normal form: the ASCII lowercase letters `a` to `z` become uppercase, and every hyphen `-` and space ` ` is
   removed. No other character is changed.
2. Each character of the normal form has a weight by its position from the left, cycling 1, 3, 7: the 1st
   character has weight 1, the 2nd 3, the 3rd 7, the 4th 1 again, and so on.
3. Echo rule: a character identical to the character immediately before it in the normal form contributes 0.
   Every other character contributes its value times its weight. An echoed character still takes its position in
   the weight cycle.
4. The check character is the alphabet character whose value is (the sum of all contributions + the number of
   characters in the normal form) modulo 28.
5. `seal` raises `ValueError` when the normal form is empty or contains any character not in the alphabet (for
   example `B`, `I`, `O`, `!`, `_`, or a tab).

`is_sealed(text: str) -> bool` returns `True` when the normal form of `text` (rule 1) has at least 2 characters,
all of them in the alphabet, and its last character is the check character that rule 4 gives for the characters
before it; otherwise it returns `False`. It never raises. Both functions are only ever given a `str`.

- `seal("ac 77")` returns `"AC77-D"`
- `is_sealed("ac77d")` returns `True`
