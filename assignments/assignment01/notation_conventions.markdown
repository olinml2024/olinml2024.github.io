---
title: "Notation Conventions"
author: "Machine Learning"
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Notation

## Scalars

We will use lower-case, unbolded letters to refer to scalar quantities. For example, we would refer to the scalar quantity `x` using the following notation.

$$
x
$$

## Vectors

We will use lower-case, bolded letters to refer to vector quantities. For example, we would refer to the vector quantity `v` using the following notation.

$$
\mathbf{v}
$$

### Vector Indexing

We will use the notation $v_i$ to refer to the i-th element of the vector $\mathbf{v}$.

### Row Versus Column Vectors

When we talk about a vector, unless otherwise specified, we will be referring to a column vector (i.e., a matrix with shape $d \times 1$).

## Matrices

We will use upper-case, bolded letters to refer to matrix quantities. For example, we would refer to the matrix quantity `A` using the following notation.

$$
\mathbf{A}
$$

### Matrix Indexing

1. We will refer to the i-th column of the matrix $\mathbf{A}$ as $\mathbf{a}_{i}$.
2. We do not currently have a shorthand to refer to the i-th row of a matrix.
3. We will refer to the element at row $i$, column $j$ of matrix $\mathbf{A}$ as $a_{i, j}$.

## Independent versus Dependent variables

We will use `x` to refer to independent (i.e., input) variables and `y` to refer to dependent (i.e., output) variables. For instance, when describing training data, we will always use `x` to refer to the input variables and `y` to refer to the output variable.
