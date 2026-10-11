# stitch-markup

`render(src: str) -> str` renders Stitch markup. Three marker characters open and close spans, and each span
transforms its rendered content:

- `*` shout: the content uppercased with `str.upper`.
- `~` mirror: the content reversed character by character.
- `^` lead: the first character of the content uppercased with `str.upper`, the rest unchanged (empty content stays
  empty).

Rules:

1. Read `src` from left to right, keeping a stack of open spans. A marker character closes the innermost open span
   when that span was opened by the same marker character; otherwise it opens a new span (even when an outer span
   was opened by that character). Marker characters that open or close spans never appear in the output.
2. A span's content is rendered first, including its inner spans, and the span's own transformation is then
   applied to that result. An empty span renders as nothing.
3. A backslash escapes the next character: backslash followed by `*`, `~`, `^` or another backslash stands for that
   one literal character. It is ordinary text: it is transformed by every span it sits in, and it never opens or
   closes a span.
4. Every other character is ordinary text and is copied through (transformed by the spans it sits in).
5. `ValueError` is raised when a backslash is followed by any other character or is the last character of `src`,
   or when any span is still open at the end of `src`. `src` is always a `str`.

- `render("say *hi* ~olleh~")` returns `"say HI hello"`
- `render("*a~bc~*")` returns `"ACB"`
