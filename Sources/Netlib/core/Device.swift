//******************************************************************************
//  Created by Edward Connell on 3/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// ComputeService
public protocol ComputeService: ObjectTracking, Logging {
    init(logging: LogInfo) throws
    var devices: [ComputeDevice] { get }
    var id: Int { get set }
    var name: String { get }
}

//==============================================================================
// ComputeDevice
//    This specifies the compute device interface
public protocol ComputeDevice: ObjectTracking, Logging {
    //-------------------------------------
    // properties
    /// a dictionary of device specific attributes describing the device
    var attributes: [String: String] { get }
    /// the amount of free memory currently available on the device
    var availableMemory: UInt64 { get }
    /// the id of the device for example gpu:0
    var id: Int { get }
    /// the maximum number of threads supported per block
    var maxThreadsPerBlock: Int { get }
    /// the name of the device
    var name: String { get }
    /// the service this device belongs to
    var service: ComputeService! { get }
    /// is `true` if the device is configured to use unified memory addressing
    /// with the host CPU
    var usesUnifiedAddressing: Bool { get }
    /// current percent of the device utilized
    var utilization: Float { get }

    //-------------------------------------
    // device resource functions
    /// creates an array on this device
    func createArray(count: Int) throws -> DeviceArray
    /// creates a named command stream for this device
    func createStream(name: String) throws -> DeviceStream
}

//==============================================================================
// DeviceArray
//    This represents a device data array
public protocol DeviceArray: ObjectTracking, Logging {
    //-------------------------------------
    // properties
    /// the device where this array is allocated
    var device: ComputeDevice { get }
    /// a pointer to the memory on the device
    var data: UnsafeMutableRawPointer { get }
    /// the size of the device memory in bytes
    var count: Int { get }
    /// the array edit version number used for replication and synchronization
    var version: Int { get set }

    //-------------------------------------
    // functions
    /// clears the array to zero
    func zero(using stream: DeviceStream?) throws
    /// asynchronously copies the contents of another device array
    func copyAsync(from other: DeviceArray, using stream: DeviceStream) throws
    /// asynchronously copies the contents of a memory buffer
    func copyAsync(from buffer: UnsafeRawBufferPointer,
                   using stream: DeviceStream) throws
    /// copies the contents to a memory buffer synchronously
    func copy(to buffer: UnsafeMutableRawBufferPointer,
              using stream: DeviceStream) throws
    /// copies the contents to a memory buffer asynchronously
    func copyAsync(to buffer: UnsafeMutableRawBufferPointer,
                   using stream: DeviceStream) throws
}

//==============================================================================
// StreamEvent
/// Stream events are queued to enable stream synchronization
public protocol StreamEvent: ObjectTracking, Logging {
    /// is `true` if the even has occurred
    var occurred: Bool { get }

    init(options: StreamEventOptions) throws
}

public struct StreamEventOptions: OptionSet {
    public init(rawValue: Int) { self.rawValue = rawValue }
    public let rawValue: Int
    /// hostSync     = StreamEventOptions(rawValue: 1 << 0)
    /// timing       = StreamEventOptions(rawValue: 1 << 1)
    /// interProcess = StreamEventOptions(rawValue: 1 << 2)
}

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

    /// max
    /// maximum
    /// mean
    /// min
    /// minimum
    /// mod
    /// mul
    func mul<T: Numeric>(lhs: TensorView<T>, rhs: TensorView<T>,
                         result: inout TensorView<T>) throws
    /// neg
    
    /// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
    /// scalars.
    /// - Note: `.==` supports broadcasting.
    func notEqual<T: Numeric>(lhs: TensorView<T>, rhs: TensorView<T>,
                              result: inout TensorView<Bool>) throws

    /// pad
    /// pow
    /// prod

    /// Computes the element-wise `rsqrt`
    func rsqrt<T: FloatingPoint>(x: TensorView<T>, result: inout TensorView<T>) throws

    /// select
    
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
