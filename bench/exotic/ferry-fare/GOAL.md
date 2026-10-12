# ferry-fare

`fare(ages: list[int], vehicle: str) -> int` returns the total fare, in crowns, for one party's crossing on the
Skelling ferry, given each passenger's age in whole years and the vehicle the party brings. Rules:

1. Each passenger pays by age: 0 to 3 pay 0; 4 to 15 pay 3; 16 to 64 pay 7; 65 and over pay 4.
2. The vehicle pays a fee: `"none"` 0, `"bike"` 2, `"car"` 12, `"van"` 18.
3. Group waiver: when 4 or more passengers have a fare above 0, the lowest of those fares is waived. Only one
   passenger's fare is waived, even when several share the lowest fare.
4. Carpool: a `"car"` carrying 3 or more passengers (of any age, counting those who pay 0) pays a fee of 6 instead
   of 12. A `"van"` never gets this reduction.
5. The total is the sum of the passenger fares after rule 3 plus the vehicle fee after rule 4.
6. `ValueError` is raised when an age is below 0 or above 120, when `vehicle` is not exactly one of the four names
   above (names are case-sensitive), when a `"car"` or `"van"` has no passenger aged 16 or over, or when a `"bike"`
   has no passengers. A party with `"none"` needs no adult, and an empty party with `"none"` costs 0.

Arguments always have the annotated types; nothing else needs checking.

- `fare([30, 8], "car")` returns `22`
- `fare([70, 40, 10, 2], "car")` returns `20`
