//******************************************************************************
//  Created by Edward Connell on 3/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// DeviceStream
/// A device stream is an asynchronous queue of commands executed on
/// the associated device
public protocol DeviceStream: ObjectTracking, Logging {
    //--------------------------------------------------------------------------
    // properties
    /// the device the stream is associated with
    var device: ComputeDevice { get }
    /// a unique id used to identify the stream
    var id: Int { get }
    /// a name used to identify the stream
    var name: String { get }

    //--------------------------------------------------------------------------
    // synchronization functions
    /// blocks the calling thread until the stream queue is empty
    func blockCallerUntilComplete() throws
    /// creates a StreamEvent
    func createEvent(options: StreamEventOptions) throws -> StreamEvent
    /// creates an artificial delay used to simulate work for debugging
    func debugDelay(seconds: Double) throws
    /// queues a stream event
    func record(event: StreamEvent) throws -> StreamEvent
    /// blocks caller until the event has occurred on this stream,
    /// then recorded and occurred on the other stream
    func sync(with other: DeviceStream, event: StreamEvent) throws
    /// blocks caller until the event has occurred
    func wait(for event: StreamEvent) throws

    //--------------------------------------------------------------------------
    // intrinsic functions
    /// Computes the absolute value of the specified TensorView element-wise.
    func abs<T>(x: T, result: inout T) throws
        where T: TensorDataView, T.Scalar: SignedNumeric

    /// Adds two tensors and produces their sum.
    /// - Note: `+` supports broadcasting.
    func add<TL, TR>(lhs: TL, rhs: TR, result: inout TL) throws where
        TL: TensorDataView, TL.Scalar: Numeric,
        TR: TensorDataView, TR.Scalar: Numeric

    /// Returns `true` if all scalars are equal to `true`. Otherwise, returns
    /// `false`.
    func all<T>(x: T, result: inout T) throws
        where T: TensorDataView, T.Scalar == Bool

    /// Returns a `true` scalar if any scalars are equal to `true`.
    /// Otherwise returns a `false` scalar
    func any<T>(x: T, result: inout T) throws
        where T: TensorDataView, T.Scalar == Bool

    /// Performs a pointwise comparison within the specified tolerance
    func approximatelyEqual<T, R>(lhs: T, rhs: T, tolerance: Double,
                                  result: inout R) throws where
        T: TensorDataView, T.Scalar: FloatingPoint,
        R: TensorDataView, R.Scalar == Bool

    /// Returns the indices of the maximum values along the specified axes. The
    /// reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func argmax<T, R>(x: T, squeezingAxis axis: Int, result: inout R) throws where
        T: TensorDataView, T.Scalar: Numeric,
        R: TensorDataView, R.Scalar == Int32

    /// Returns the indices of the minimum values along the specified axes. The
    /// reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func argmin<T, R>(x: T, squeezingAxis axis: Int, result: inout R) throws where
        T: TensorDataView, T.Scalar: Numeric,
        R: TensorDataView, R.Scalar == Int32

    /// Broadcast x to the specified shape
    /// - Parameter x: the pattern to broadcast
    /// - Parameter shape: the shape of the result
    /// - Precondition: The specified shape must be compatible for broadcasting.
    func broadcast<T>(x: T, toShape shape: DataShape, result: inout T) throws
        where T: TensorDataView

    /// cast scalar types
    /// - Parameter from: the input data
    /// - Parameter result: the output
    func cast<T, R>(from: T, to result: inout R) throws where
        T: TensorDataView, T.Scalar: AnyNumeric,
        R: TensorDataView, R.Scalar: AnyNumeric

    /// Computes the ceiling of the specified TensorView element-wise.
    func ceil<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: FloatingPoint

    /// Concatenates tensors along the specified axis.
    /// - Precondition: The tensors must have the same dimensions, except for the
    ///                 specified axis.
    /// - Precondition: The axis must be in the range `-rank..<rank`.
    func concatenate<T>(view: T, with other: T, alongAxis axis: Int,
                        result: inout T) throws where
        T: TensorDataView

    /// Computes the element-wise `cos`
    func cos<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: FloatingPoint

    /// Computes the element-wise `cosh`
    func cosh<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: FloatingPoint

    /// Returns the quotient of dividing the first TensorView by the second.
    /// - Note: `/` supports broadcasting.
    func div<TL, TR>(lhs: TL, rhs: TR, result: inout TL) throws where
        TL: TensorDataView, TL.Scalar: Numeric,
        TR: TensorDataView, TR.Scalar: Numeric

    /// Computes `lhs == rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    /// - Note: `.==` supports broadcasting.
    func equal<T, R>(lhs: T, rhs: T, result: inout R) throws where
        T: TensorDataView, T.Scalar: Numeric,
        R: TensorDataView, R.Scalar == Bool

    /// Computes the element-wise `exp`
    func exp<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: FloatingPoint

    /// Computes the element-wise `floor`
    func floor<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: FloatingPoint

    /// Computes `lhs > rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    func greater<T, R>(lhs: T, rhs: T, result: inout R) throws where
        T: TensorDataView, T.Scalar: Numeric & Comparable,
        R: TensorDataView, R.Scalar == Bool

    /// Computes `lhs >= rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    func greaterOrEqual<T, R>(lhs: T, rhs: T, result: inout R) throws where
        T: TensorDataView, T.Scalar: Numeric & Comparable,
        R: TensorDataView, R.Scalar == Bool

    /// Computes `lhs < rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    func less<T, R>(lhs: T, rhs: T, result: inout R) throws where
        T: TensorDataView, T.Scalar: Numeric & Comparable,
        R: TensorDataView, R.Scalar == Bool
    
    /// lessEqual
    /// Computes `lhs <= rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    func lessOrEqual<T, R>(lhs: T, rhs: T, result: inout R) throws where
        T: TensorDataView, T.Scalar: Numeric & Comparable,
        R: TensorDataView, R.Scalar == Bool

    /// Computes the element-wise `log`
    func log<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: FloatingPoint
    
    /// Computes the element-wise `!x`
    func logicalNot<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar == Bool
    
    /// Computes the element-wise `lhs && rhs`
    func logicalAnd<T>(lhs: T, rhs: T, result: inout T) throws where
        T: TensorDataView, T.Scalar == Bool
    
    /// Computes the element-wise `lhs || rhs`
    func logicalOr<T>(lhs: T, rhs: T, result: inout T) throws where
        T: TensorDataView, T.Scalar == Bool

    /// Computes the element-wise `logSoftmax`
    func logSoftmax<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: FloatingPoint

    /// Performs matrix multiplication with another TensorView and produces the
    /// result.
    func matmul<T>(lhs: T, rhs: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: Numeric

    /// Returns the maximum values along the specified axes. The reduced
    /// dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func max<T>(x: T, squeezingAxes axes: [Int], result: inout T) throws where
        T: TensorDataView, T.Scalar: Numeric
    
    /// Computes the element-wise maximum of two tensors.
    /// - Note: `max` supports broadcasting.
    func maximum<T>(lhs: T, rhs: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: Numeric

    /// Returns the arithmetic mean along the specified axes. The reduced
    /// dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    func mean<T>(x: T, squeezingAxes axes: [Int], result: inout T) throws where
        T: TensorDataView, T.Scalar: Numeric

    /// Returns the minimum values along the specified axes. The reduced
    /// dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func min<T>(x: T, squeezingAxes axes: [Int], result: inout T) throws where
        T: TensorDataView, T.Scalar: Numeric
    
    /// Computes the element-wise minimum of two tensors.
    /// - Note: `max` supports broadcasting.
    func minimum<T>(lhs: T, rhs: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: Numeric

    /// Returns the remainder of dividing the first TensorView by the second.
    /// - Note: `%` supports broadcasting.
    func mod<TL, TR>(lhs: TL, rhs: TR, result: inout TL) throws where
        TL: TensorDataView, TL.Scalar: Numeric,
        TR: TensorDataView, TR.Scalar: Numeric

    /// mul
    func mul<TL, TR>(lhs: TL, rhs: TR, result: inout TL) throws where
        TL: TensorDataView, TL.Scalar: Numeric,
        TR: TensorDataView, TR.Scalar: Numeric

    /// Computes the element-wise negation
    func neg<T>(x: T, result: inout T) throws where
    T: TensorDataView, T.Scalar: SignedNumeric

    /// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    /// - Note: `.==` supports broadcasting.
    func notEqual<TL, TR, R>(lhs: TL, rhs: TR, result: inout R) throws where
        TL: TensorDataView, TL.Scalar: Numeric & Comparable,
        TR: TensorDataView, TR.Scalar: Numeric & Comparable,
        R: TensorDataView, R.Scalar == Bool
    
    /// Returns a padded TensorView according to the specified margins.
    /// TODO: Probably don't need this. It can be accomplished via TensorView indexing
    func pad<T>(x: T, with margins: [(before: Int, after: Int)],
                fillValue: T.Scalar, result: inout T) throws where
        T: TensorDataView
    
    /// Computes the element-wise `x**y`
    func pow<TX, TY, TR>(x: TX, y: TY, result: inout TR) throws where
        TX: TensorDataView, TX.Scalar: Numeric,
        TY: TensorDataView, TY.Scalar: Numeric,
        TR: TensorDataView, TR.Scalar: Numeric

    /// Product of the input elements to produce a scalar
    func prod<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: Numeric

    /// Computes the element-wise `rsqrt`
    func rsqrt<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: FloatingPoint

    /// Replaces elements of `x` with `other` in the lanes where `mask` is`true`
    ///
    /// - Precondition: `x` and `other` must have the same shape. If
    ///   `x` and `other` are scalar, then `mask` must also be scalar. If
    ///   `x` and `other` have rank greater than or equal to `1`, then `mask`
    ///   must be either have the same shape as `self` or be a 1-D `TensorView` such
    ///   that `mask.scalarCount == self.shape[0]`.
    func replacing<T, R>(x: T, with other: T, where mask: R,
                      result: inout T) throws where
        T: TensorDataView,
        R: TensorDataView, R.Scalar == Bool
    
    /// Computes the element-wise `sin`
    func sin<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: FloatingPoint
    
    /// Computes the element-wise `sinh`
    func sinh<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: FloatingPoint
    
    /// Computes the element-wise `square`
    func square<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: Numeric

    /// Computes the element-wise `(lhs - rhs)**2`
    func squaredDifference<T>(lhs: T, rhs: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: Numeric

    /// Computes the element-wise `sqrt`
    func sqrt<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: FloatingPoint

    /// subtract
    func subtract<TL, TR>(lhs: TL, rhs: TR, result: inout TL) throws where
        TL: TensorDataView, TL.Scalar: Numeric,
        TR: TensorDataView, TR.Scalar: Numeric

    /// Sums the input to produce a scalar
    func sum<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: Numeric

    /// Computes the element-wise `tan`
    func tan<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: FloatingPoint
    
    /// Computes the element-wise `tanh`
    func tanh<T>(x: T, result: inout T) throws where
        T: TensorDataView, T.Scalar: FloatingPoint
}
