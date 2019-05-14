//******************************************************************************
//  Created by Edward Connell on 3/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// DeviceStream
/// A device stream is an asynchronous queue of commands executed on
/// the associated device. It is a class protocol treated as an abstract
/// driver interface.
public protocol DeviceStream:
    ObjectTracking,
    Logger,
    DeviceErrorHandling,
    StreamIntrinsicsProtocol,
    StreamGradientsProtocol
{
    //--------------------------------------------------------------------------
    /// the device the stream is associated with
    var device: ComputeDevice { get }
    /// if `true` the stream will execute functions synchronous with the app
    /// it is `false` by default and used for debugging
    var executeSynchronously: Bool { get set }
    /// a unique id used to identify the stream
    var id: Int { get }
    /// a name used to identify the stream
    var name: String { get }
    /// the maximum time to wait for an operation to complete
    /// a value of 0 (default) will wait forever
    var timeout: TimeInterval { get set }
    
    //--------------------------------------------------------------------------
    // synchronization functions
    /// creates a StreamEvent
    func createEvent(options: StreamEventOptions) throws -> StreamEvent
    /// queues a stream event op. When executed the event is signaled
    @discardableResult
    func record(event: StreamEvent) throws -> StreamEvent
    /// records an op on the stream that will perform a stream blocking wait
    /// when it is processed
    func wait(for event: StreamEvent) throws
    /// blocks the calling thread until the stream queue has completed all work
    func waitUntilStreamIsComplete() throws

    //--------------------------------------------------------------------------
    // debugging functions
    /// simulateWork(x:timePerElement:result:
    /// introduces a delay in the stream by sleeping a duration of
    /// x.shape.elementCount * timePerElement
    func simulateWork<T>(x: T, timePerElement: TimeInterval, result: inout T)
        where T: TensorView
    /// causes the stream to sleep for the specified interval for testing
    func delayStream(atLeast interval: TimeInterval)
    /// for unit testing. It's part of the class protocol so that remote
    /// streams throw the error remotely.
    func throwTestError()
}

public extension DeviceStream {
    func createEvent() throws -> StreamEvent {
        return try createEvent(options: StreamEventOptions())
    }
}

let streamThreadViolationMessage =
"a stream can only be accessed by the thread that created it"

//==============================================================================
/// LocalDeviceStream
public protocol LocalDeviceStream: DeviceStream { }

public extension LocalDeviceStream {
    //--------------------------------------------------------------------------
    /// handleDevice(error:
    func handleDevice(error: Error) {
        if (deviceErrorHandler?(error) ?? .propagate) == .propagate {
            device.handleDevice(error: error)
        }
    }
}

//==============================================================================
// StreamIntrinsicsProtocol
/// The required set of base level intrinsic functions for a `DeviceStream`
///
public protocol StreamIntrinsicsProtocol {
    /// Computes the absolute value of the specified TensorView element-wise.
    func abs<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: Numeric, T.Stored.Magnitude == T.Stored
    /// Adds two tensors and produces their sum.
    func add<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Stored: Numeric
    /// Returns `true` if all scalars are `true`. Otherwise, returns `false`.
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    func all<T>(x: T, along axes: Vector<IndexScalar>?, result: inout T)
        where T: TensorView, T.Stored == Bool
    /// Returns `true` if any scalars are`true`. Otherwise, returns `false`.
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    func any<T>(x: T, along axes: Vector<IndexScalar>?, result: inout T)
        where T: TensorView, T.Stored == Bool
    /// Performs a pointwise comparison within the specified tolerance
    func approximatelyEqual<T>(lhs: T, rhs: T,
                               tolerance: T.Stored,
                               result: inout T.BoolView) where
        T: TensorView, T.Stored: AnyFloatingPoint,
        T.BoolView.Stored == Bool
    /// Returns the indices of the maximum values along the specified axes. The
    /// reduced dimensions are removed.
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func argmax<T>(x: T, along axes: Vector<IndexScalar>?,
                   result: inout T.IndexView) where
        T: TensorView, T.Stored: Numeric,
        T.IndexView.Stored == IndexScalar
    /// Returns the indices of the minimum values along the specified axes. The
    /// reduced dimensions are removed.
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func argmin<T>(x: T, along axes: Vector<IndexScalar>?,
                   result: inout T.IndexView) where
        T: TensorView, T.Stored: Numeric,
        T.IndexView.Stored == IndexScalar
    /// Sums the absolute value of the input along the specified axes
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    func asum<T>(x: T, along axes: Vector<IndexScalar>?, result: inout T) where
        T: TensorView, T.Stored: Numeric,
        T.Stored.Magnitude == T.Stored
    /// cast scalar types
    /// - Parameter from: the input data
    /// - Parameter result: the output
    func cast<T, R>(from: T, to result: inout R) where
        T: TensorView, T.Stored: AnyConvertable,
        R: TensorView, R.Stored: AnyConvertable
    /// Computes the ceiling of the specified TensorView element-wise.
    func ceil<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: AnyFloatingPoint
    /// Concatenates tensors along the specified axis.
    /// - Precondition: The tensors must have the same dimensions, except for the
    ///                 specified axis.
    /// - Precondition: The axis must be in the range `-rank..<rank`.
    func concatenate<T>(view: T, with other: T, alongAxis axis: Int,
                        result: inout T) where T: TensorView
    /// Computes the element-wise `cos`
    func cos<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: AnyFloatingPoint
    /// Computes the element-wise `cosh`
    func cosh<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: AnyFloatingPoint
    /// Returns the quotient of dividing the first TensorView by the second.
    /// - Note: `/` supports broadcasting.
    func div<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Stored: AnyFloatingPoint
    /// Computes `lhs == rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    /// - Note: `.==` supports broadcasting.
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Stored: Equatable,
        T.BoolView.Stored == Bool
    /// Computes the element-wise `exp`
    func exp<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: AnyFloatingPoint
    /// fills the view with the scalar value
    func fill<T>(_ result: inout T, with: T.Stored) where T: TensorView
    /// fills the view with the spatial sequential index
    func fillWithIndex<T>(_ result: inout T, startAt: Int) where
        T: TensorView, T.Stored: AnyNumeric
    /// Computes the element-wise `floor`
    func floor<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: AnyFloatingPoint
    /// Computes `lhs > rhs` element-wise and returns a tensor of Bool scalars.
    func greater<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Stored: Comparable, T.BoolView.Stored == Bool
    /// Computes `lhs >= rhs` element-wise and returns a tensor of Bool scalars.
    func greaterOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Stored: Comparable, T.BoolView.Stored == Bool
    /// Computes `lhs < rhs` element-wise and returns a tensor of Bool scalars.
    func less<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Stored: Comparable, T.BoolView.Stored == Bool
    /// Computes `lhs <= rhs` element-wise and returns a tensor of Bool scalars.
    func lessOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Stored: Comparable, T.BoolView.Stored == Bool
    /// Computes the element-wise `log`
    func log<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: AnyFloatingPoint
    /// Computes the element-wise `!x`
    func logicalNot<T>(x: T, result: inout T) where
        T: TensorView, T.Stored == Bool
    /// Computes the element-wise `lhs && rhs`
    func logicalAnd<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Stored == Bool
    /// Computes the element-wise `lhs || rhs`
    func logicalOr<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Stored == Bool
    /// Computes the element-wise `logSoftmax`
    func logSoftmax<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: AnyFloatingPoint
    /// Performs matrix multiplication with another TensorView and produces the
    /// result.
    func matmul<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Stored: Numeric
    /// Returns the maximum values along the specified axes. The reduced
    /// dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func max<T>(x: T, along axes: [Int], result: inout T) where
        T: TensorView, T.Stored: Numeric
    /// Computes the element-wise maximum of two tensors.
    /// - Note: `max` supports broadcasting.
    func maximum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Stored: Comparable
    /// Returns the arithmetic mean along the specified axes. The reduced
    /// dimensions are removed.
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    func mean<T>(x: T, along axes: Vector<IndexScalar>?, result: inout T) where
        T: TensorView, T.Stored: Numeric
    /// Returns the minimum values along the specified axes. The reduced
    /// dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func min<T>(x: T, along axes: [Int], result: inout T) where
        T: TensorView, T.Stored: Numeric
    /// Computes the element-wise minimum of two tensors.
    /// - Note: `max` supports broadcasting.
    func minimum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Stored: Comparable
    /// Returns the remainder of dividing the first TensorView by the second.
    /// - Note: `%` supports broadcasting.
    func mod<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Stored: AnyFloatingPoint
    /// mul
    func mul<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Stored: Numeric
    /// Computes the element-wise negation
    func neg<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: SignedNumeric
    /// Computes `lhs != rhs` element-wise and returns a tensor of Bools
    /// - Note: `.==` supports broadcasting.
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Stored: Numeric, T.BoolView.Stored == Bool
    /// Computes the element-wise `x**y`
    func pow<T>(x: T, y: T, result: inout T)
        where T: TensorView, T.Stored: AnyNumeric
    /// Product of the input elements to produce a scalar
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
    func prod<T>(x: T, along axes: Vector<IndexScalar>?, result: inout T) where
        T: TensorView, T.Stored: AnyNumeric
    /// Computes the element-wise `rsqrt`
    func rsqrt<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: AnyFloatingPoint
    /// Replaces elements of `x` with `other` in the lanes where `mask` is`true`
    ///
    /// - Precondition: `x` and `other` must have the same shape. If
    ///   `x` and `other` are scalar, then `mask` must also be scalar. If
    ///   `x` and `other` have rank greater than or equal to `1`, then `mask`
    ///   must be either have the same shape as `self` or be a 1-D `TensorView` such
    ///   that `mask.scalarCount == self.shape[0]`.
    func replacing<T>(x: T, with other: T, where mask: T.BoolView,
                      result: inout T)
        where T: TensorView, T.BoolView.Stored == Bool
    /// Computes the element-wise `sin`
    func sin<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: AnyFloatingPoint
    /// Computes the element-wise `sinh`
    func sinh<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: AnyFloatingPoint
    /// Computes the element-wise `square`
    func square<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: Numeric
    /// Computes the element-wise `(lhs - rhs)**2`
    func squaredDifference<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Stored: Numeric
    /// Computes the element-wise `sqrt`
    func sqrt<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: AnyFloatingPoint
    /// subtract
    func subtract<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Stored: Numeric
    /// Sums the input along the specified axes
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    func sum<T>(x: T, along axes: Vector<IndexScalar>?, result: inout T) where
        T: TensorView, T.Stored: Numeric
    /// Computes the element-wise `tan`
    func tan<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: AnyFloatingPoint
    /// Computes the element-wise `tanh`
    func tanh<T>(x: T, result: inout T) where
        T: TensorView, T.Stored: AnyFloatingPoint
}
