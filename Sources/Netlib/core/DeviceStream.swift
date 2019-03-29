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
    func abs<T: SignedNumeric>(x: TensorView<T>, result: inout TensorView<T>) throws

    /// Adds two tensors and produces their sum.
    /// - Note: `+` supports broadcasting.
    func add<T: Numeric>(lhs: TensorView<T>, rhs: TensorView<T>,
                         result: inout TensorView<T>) throws

    /// Returns `true` if all scalars are equal to `true`. Otherwise, returns
    /// `false`.
    func all(x: TensorView<Bool>, result: inout TensorView<Bool>) throws

    /// Returns a `true` scalar if any scalars are equal to `true`.
    /// Otherwise returns a `false` scalar
    func any(x: TensorView<Bool>, result: inout TensorView<Bool>) throws

    /// Performs a pointwise comparison within the specified tolerance
    func approximatelyEqual<T: FloatingPoint>(
        lhs: TensorView<T>, rhs: TensorView<T>,
        tolerance: Double,
        result: inout TensorView<Bool>) throws
    
    /// Returns the indices of the maximum values along the specified axes. The
    /// reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func argmax<T: Numeric>(x: TensorView<T>, squeezingAxis axis: Int,
                            result: inout TensorView<Int32>) throws

    /// Returns the indices of the minimum values along the specified axes. The
    /// reduced dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func argmin<T: Numeric>(x: TensorView<T>, squeezingAxis axis: Int,
                            result: inout TensorView<Int32>) throws

    /// Broadcast x to the specified shape
    /// - Parameter x: the pattern to broadcast
    /// - Parameter shape: the shape of the result
    /// - Precondition: The specified shape must be compatible for broadcasting.
    func broadcast<T>(x: TensorView<T>, toShape shape: DataShape,
                      result: inout TensorView<T>) throws
    /// cast scalar types
    /// - Parameter from: the input data
    /// - Parameter result: the output
    func cast<T,U>(from: TensorView<T>, to result: inout TensorView<U>) throws
    
    /// Computes the ceiling of the specified TensorView element-wise.
    func ceil<T : FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws

    /// Concatenates tensors along the specified axis.
    /// - Precondition: The tensors must have the same dimensions, except for the
    ///                 specified axis.
    /// - Precondition: The axis must be in the range `-rank..<rank`.
    func concatenate<T>(view: TensorView<T>, with other: TensorView<T>,
                        alongAxis axis: Int, result: inout TensorView<T>) throws
    
    /// Computes the element-wise `cos`
    func cos<T: FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws
    
    /// Computes the element-wise `cosh`
    func cohs<T: FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws

    /// Returns the quotient of dividing the first TensorView by the second.
    /// - Note: `/` supports broadcasting.
    func div<T: Numeric>(lhs: TensorView<T>, rhs: TensorView<T>,
                         result: inout TensorView<T>) throws
    
    /// Computes `lhs == rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    /// - Note: `.==` supports broadcasting.
    func equal<T: Numeric>(lhs: TensorView<T>, rhs: TensorView<T>,
                           result: inout TensorView<Bool>) throws

    /// Computes the element-wise `exp`
    func exp<T: FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws

    /// Computes the element-wise `floor`
    func floor<T : FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws

    /// Computes `lhs > rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    func greater<T: Numeric & Comparable>(
        lhs: TensorView<T>, rhs: TensorView<T>,
        result: inout TensorView<Bool>) throws

    /// Computes `lhs >= rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    func greaterOrEqual<T: Numeric & Comparable>(
        lhs: TensorView<T>, rhs: TensorView<T>,
        result: inout TensorView<Bool>) throws

    /// Computes `lhs < rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    func less<T: Numeric & Comparable>(
        lhs: TensorView<T>, rhs: TensorView<T>,
        result: inout TensorView<Bool>) throws
    
    /// lessEqual
    /// Computes `lhs <= rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    func lessOrEqual<T: Numeric & Comparable>(
        lhs: TensorView<T>, rhs: TensorView<T>,
        result: inout TensorView<Bool>) throws

    /// Computes the element-wise `log`
    func log<T: FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws
    
    /// Computes the element-wise `!x`
    func logicalNot(x: TensorView<Bool>, result: inout TensorView<Bool>) throws
    
    /// Computes the element-wise `lhs && rhs`
    func logicalAnd(lhs: TensorView<Bool>, rhs: TensorView<Bool>,
                    result: inout TensorView<Bool>) throws
    
    /// Computes the element-wise `lhs || rhs`
    func logicalOr(lhs: TensorView<Bool>, rhs: TensorView<Bool>,
                   result: inout TensorView<Bool>) throws

    /// Computes the element-wise `logSoftmax`
    func logSoftmax<T: FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws

    /// Performs matrix multiplication with another TensorView and produces the
    /// result.
    func matmul<T: Numeric>(lhs: TensorView<T>, rhs: TensorView<T>,
                            result: inout TensorView<T>) throws

    /// Returns the maximum values along the specified axes. The reduced
    /// dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func max<T: Numeric>(x: TensorView<T>, squeezingAxes axes: [Int],
                         result: inout TensorView<T>) throws
    
    /// Computes the element-wise maximum of two tensors.
    /// - Note: `max` supports broadcasting.
    func maximum<T: Numeric>(lhs: TensorView<T>, rhs: TensorView<T>,
                             result: inout TensorView<T>) throws

    /// Returns the arithmetic mean along the specified axes. The reduced
    /// dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    func mean<T: Numeric>(x: TensorView<T>, squeezingAxes axes: [Int],
                          result: inout TensorView<T>) throws

    /// Returns the minimum values along the specified axes. The reduced
    /// dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func min<T: Numeric>(x: TensorView<T>, squeezingAxes axes: [Int],
                         result: inout TensorView<T>) throws
    
    /// Computes the element-wise minimum of two tensors.
    /// - Note: `max` supports broadcasting.
    func minimum<T: Numeric>(lhs: TensorView<T>, rhs: TensorView<T>,
                             result: inout TensorView<T>) throws

    /// Returns the remainder of dividing the first TensorView by the second.
    /// - Note: `%` supports broadcasting.
    func mod<T: Numeric>(lhs: TensorView<T>, rhs: TensorView<T>,
                         result: inout TensorView<T>) throws
    /// mul
    func mul<T: Numeric>(lhs: TensorView<T>, rhs: TensorView<T>,
                         result: inout TensorView<T>) throws

    /// Computes the element-wise negation
    func neg(x: TensorView<Bool>, result: inout TensorView<Bool>) throws

    /// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    /// - Note: `.==` supports broadcasting.
    func notEqual<T: Numeric>(lhs: TensorView<T>, rhs: TensorView<T>,
                              result: inout TensorView<Bool>) throws

    /// Returns a padded TensorView according to the specified margins.
    /// TODO: Probably don't need this. It can be accomplished via TensorView indexing
    func pad<T>(x: TensorView<T>,
                with margins: [(before: Int32, after: Int32)],
                fillValue: T,
                result: inout TensorView<T>) throws
    
    /// Computes the element-wise `x**y`
    func pow<T: Numeric>(x: TensorView<T>, y: TensorView<T>,
                                       result: inout TensorView<T>) throws

    /// Product of the input elements to produce a scalar
    func prod<T: Numeric>(x: TensorView<T>, result: inout TensorView<T>) throws

    /// Computes the element-wise `rsqrt`
    func rsqrt<T: FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws

    /// Replaces elements of `x` with `other` in the lanes where `mask` is`true`
    ///
    /// - Precondition: `x` and `other` must have the same shape. If
    ///   `x` and `other` are scalar, then `mask` must also be scalar. If
    ///   `x` and `other` have rank greater than or equal to `1`, then `mask`
    ///   must be either have the same shape as `self` or be a 1-D `TensorView` such
    ///   that `mask.scalarCount == self.shape[0]`.
    func replacing<T>(x: TensorView<T>, with other: TensorView<T>,
                      where mask: TensorView<Bool>,
                      result: inout TensorView<T>) throws
    
    /// Computes the element-wise `sin`
    func sin<T: FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws
    
    /// Computes the element-wise `sinh`
    func sinh<T: FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws
    
    /// Computes the element-wise `square`
    func square<T: Numeric>(x: TensorView<T>, result: inout TensorView<T>) throws

    /// Computes the element-wise `(lhs - rhs)**2`
    func squaredDifference<T: Numeric>(lhs: TensorView<T>, rhs: TensorView<T>,
                                       result: inout TensorView<T>) throws

    /// Computes the element-wise `sqrt`
    func sqrt<T: FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws

    /// subtract
    func subtract<T: Numeric>(lhs: TensorView<T>, rhs: TensorView<T>,
                              result: inout TensorView<T>) throws
    
    /// Sums the input to produce a scalar
    func sum<T: Numeric>(x: TensorView<T>, result: inout TensorView<T>) throws

    /// Computes the element-wise `tan`
    func tan<T: FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws
    
    /// Computes the element-wise `tanh`
    func tanh<T: FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws
}
