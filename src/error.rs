use std::error::Error;
use std::fmt;

/// Identifies one input of a matrix product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixOperand {
    /// Left-hand matrix.
    Left,
    /// Right-hand matrix.
    Right,
}

/// Arithmetic operation that overflowed in a checked kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArithmeticOperation {
    /// Multiplication of two matrix entries.
    Multiply,
    /// Addition into an output accumulator.
    Add,
}

impl fmt::Display for ArithmeticOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Multiply => f.write_str("multiplication"),
            Self::Add => f.write_str("addition"),
        }
    }
}

impl fmt::Display for MatrixOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Left => f.write_str("left"),
            Self::Right => f.write_str("right"),
        }
    }
}

/// Errors reported by fallible multiplication and interoperability APIs.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpGemmError {
    /// The inner dimensions of the two matrices do not agree.
    DimensionMismatch {
        /// Shape of the left-hand matrix.
        left: (usize, usize),
        /// Shape of the right-hand matrix.
        right: (usize, usize),
    },
    /// A zero-copy `sprs` adapter received column-compressed storage.
    NonCsrStorage {
        /// Operand using unsupported storage.
        operand: MatrixOperand,
    },
    /// A `usize` index cannot be represented by an ecosystem index type.
    IndexOverflow {
        /// Value that could not be converted.
        value: usize,
        /// Destination index type or buffer.
        target: &'static str,
    },
    /// An ecosystem-native output rejected generated CSR buffers.
    InvalidOutputStructure(String),
    /// Scalar arithmetic overflowed while accumulating an output entry.
    ArithmeticOverflow {
        /// Operation that could not be represented by the scalar type.
        operation: ArithmeticOperation,
        /// Output row being computed.
        row: usize,
        /// Inner-dimension index of the candidate product.
        inner: usize,
        /// Output column being computed.
        column: usize,
    },
    /// A configured output-size budget would be exceeded.
    OutputNnzLimitExceeded {
        /// Maximum number of entries allowed in the output.
        limit: usize,
        /// Number of entries required after completing the current row.
        attempted: usize,
    },
}

impl fmt::Display for SpGemmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { left, right } => write!(
                f,
                "incompatible matrix dimensions: left is {}x{}, right is {}x{}",
                left.0, left.1, right.0, right.1
            ),
            Self::NonCsrStorage { operand } => {
                write!(f, "{operand} sprs operand must use CSR storage")
            }
            Self::IndexOverflow { value, target } => {
                write!(f, "index {value} cannot be represented by {target}")
            }
            Self::InvalidOutputStructure(reason) => {
                write!(f, "generated CSR output is invalid: {reason}")
            }
            Self::ArithmeticOverflow {
                operation,
                row,
                inner,
                column,
            } => write!(
                f,
                "arithmetic overflow during {operation} at output ({row}, {column}) through inner index {inner}"
            ),
            Self::OutputNnzLimitExceeded { limit, attempted } => write!(
                f,
                "sparse product requires at least {attempted} output entries, exceeding limit {limit}"
            ),
        }
    }
}

impl Error for SpGemmError {}
