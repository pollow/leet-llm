# 001 — Array Basics: Reshape & Transpose

**Level 0 · NumPy Foundations**

## Description

The two most common ways to reorganize an array — without changing its data — are
`reshape` (reinterpret the same numbers under a new shape) and `transpose` (reorder the
axes). Almost every operation later in this course is built on getting these right, and
the #1 source of bugs is mixing them up.

In this task you'll implement a **grouping round-trip**: split the last axis of an array
into several equal groups, rearrange so the group axis comes first, and then undo it
exactly. This "split a dimension into groups and process each group in parallel" pattern
shows up all over numerical computing.

You may use **only** `numpy` — no deep-learning libraries. No Python `for`-loops over the
axes; use `reshape`/`transpose` so it stays fully vectorized.

## The Math

Work with a 3-D array `x` of shape `(B, L, F)` and a group count `G` that divides `F`,
with `f = F / G`:

- **`group_last_axis(x, G)`** maps `(B, L, F)` to `(B, G, L, f)` while preserving element
  order within each grouped chunk.
- **`ungroup_last_axis(x)`** is the inverse map from `(B, G, L, f)` back to `(B, L, F)`.

`ungroup_last_axis(group_last_axis(x, G))` must equal the original `x`.

Shape constraint: `F % G == 0`.

## Function Signature

```python
def group_last_axis(x: np.ndarray, n_groups: int) -> np.ndarray: ...
def ungroup_last_axis(x: np.ndarray) -> np.ndarray: ...
```

## Read More

- NumPy — [`reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html),
  [`transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html),
  [`swapaxes`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html)
- NumPy — [memory layout & strides](https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray)
  (why reshape is free but a transpose changes the axis order)

## How to Test

```bash
uv run grade 001          # or: uv run pytest 001_numpy_array_basics
```
