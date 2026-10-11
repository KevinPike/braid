# orchard-score

`score(grid: list[str]) -> int` scores a finished orchard in the board game Bramble. Each string is one row and
each character one tile: `A` apple, `P` pear, `W` wasp, `.` empty. Two tiles are neighbours when they share an
edge (up, down, left or right; never diagonally). Rules:

1. An apple scores 1.
2. A pear scores 3, or 1 when it is crowded: it has at least one pear as a neighbour.
3. A wasp spoils every apple and pear that is its neighbour. A spoiled fruit scores 0, whatever rules 1 and 2 say.
   A spoiled pear still crowds its neighbouring pears. Wasps and empty tiles score 0.
4. Row bonus: each row that contains no `.` and no `W` scores 2 extra, even when some of its fruit is spoiled.
5. The score is the sum of all tile scores and row bonuses.
6. `ValueError` is raised when the grid has no rows, a row is the empty string, rows differ in length, or a tile is
   any character other than `A`, `P`, `W` and `.`. The grid is always a list of `str`; nothing else needs checking.

- `score(["AP.", "PW."])` returns `1`
- `score(["APA", "PPA"])` returns `10`
