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

- **`group_last_axis(x, G)`** → shape `(B, G, L, f)`.
  1. View the last axis `F` as two axes `(G, f)`: reshape `(B, L, F)` → `(B, L, G, f)`.
  2. Move the new group axis `G` to the front (just after `B`): transpose axes
     `(0, 2, 1, 3)` → `(B, G, L, f)`.

- **`ungroup_last_axis(x)`** → the exact inverse. Given `(B, G, L, f)`:
  1. Transpose back: `(0, 2, 1, 3)` → `(B, L, G, f)`.
  2. Reshape to merge the last two axes → `(B, L, G·f)`.

`ungroup_last_axis(group_last_axis(x, G))` must equal the original `x`.

> ⚠️ A reshape **without** the transpose would interleave the groups in the wrong order.
> The transpose is what makes each group a contiguous block along the leading axes.

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
