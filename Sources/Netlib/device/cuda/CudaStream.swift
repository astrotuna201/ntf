//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//

import Foundation
import Cuda

//import CudaKernels

public final class CudaStream: LocalDeviceStream, StreamGradients {
    // protocol properties
    public private(set) var trackingId = 0
    public var defaultStreamEventOptions = StreamEventOptions()
    public var device: ComputeDevice {
        return cudaDevice
    }
    public let id: Int
    public let name: String
    public var logInfo: LogInfo
    public var timeout: TimeInterval?
    public var executeSynchronously: Bool = false
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error?
    public var _errorMutex: Mutex = Mutex()

    /// used to detect accidental stream access by other threads
    private let creatorThread: Thread
    public let cudaDevice: CudaDevice
    public let handle: cudaStream_t
    public let cudnn: CudnnHandle
    public let cublas: CublasHandle

    //--------------------------------------------------------------------------
    // initializers
    public init(logInfo: LogInfo, device: CudaDevice,
                name: String, id: Int, isStatic: Bool) throws {
        // create a completion event
        cudaDevice = device
        self.logInfo = logInfo
        self.id = id
        self.name = name
        self.creatorThread = Thread.current
        let path = logInfo.namePath

        // select the specified device
        try cudaDevice.select()
        // create a stream associated with the device
        let flags = UInt32(cudaStreamNonBlocking)
        var cudaStream: cudaStream_t?
        try cudaCheck(status: cudaStreamCreateWithFlags(&cudaStream, flags))
        handle = cudaStream!
        cudnn = try CudnnHandle(deviceId: cudaDevice.id, using: handle,
                                isStatic: isStatic)
        cublas = try CublasHandle(deviceId: cudaDevice.id, using: handle,
                                  isStatic: isStatic)
        trackingId = ObjectTracker.global.register(self, namePath: path,
                                                   isStatic: isStatic)

        diagnostic("\(createString) DeviceStream(\(trackingId)) " +
                           "\(device.name)_\(name)", categories: .streamAlloc)
    }

    //--------------------------------------------------------------------------
    // deinit
    deinit {
        assert(Thread.current === creatorThread,
               "Stream has been captured and is being released by a " + 
               "different thread. Probably by a queued function on the stream.")

        diagnostic("\(releaseString) DeviceStream(\(trackingId)) " +
                           "\(device.name)_\(name)", categories: [.streamAlloc])

        do {
            // select the device
            try cudaDevice.select()

            // make sure pending queued commands complete
            // before releasing the queue
            try waitUntilStreamIsComplete()

            // release the stream
            try cudaCheck(status: cudaStreamDestroy(handle))

            // remove from object tracking
            ObjectTracker.global.remove(trackingId: trackingId)
        } catch {
            writeLog(String(describing: error))
        }

        diagnostic("\(releaseString) \(name)", categories: .streamAlloc)
    }

    //--------------------------------------------------------------------------
    // createEvent
    public func createEvent(options: StreamEventOptions) throws -> StreamEvent {
        try cudaDevice.select()
        return try CudaStreamEvent(options: options, timeout: timeout)
    }

//    //--------------------------------------------------------------------------
//    // delay the stream for event testing
//    public func delay(seconds: Double) throws {
//        let clockRate = (device as! CudaDevice).props.clockRate
//        try cudaCheck(status: cudaDelayStream(seconds, clockRate, handle))
//    }

    //----------------------------------------------------------------------------
    // record
    public func record(event: StreamEvent) throws -> StreamEvent {
        diagnostic("\(recordString) \(name) recording " +
                           "StreamEvent(\(event.trackingId))",
                   categories: .streamSync)
        try cudaDevice.select()
        let event = event as! CudaStreamEvent

        // set event time
        if defaultStreamEventOptions.contains(.timing) {
            event.recordedTime = Date()
        }

        try cudaCheck(status: cudaEventRecord(event.handle, handle))
        return event
    }

    //--------------------------------------------------------------------------
    // wait(for event
    public func wait(for event: StreamEvent) throws {
        assert(Thread.current === creatorThread, streamThreadViolationMessage)
        diagnostic("\(waitString) \(name) waiting for " +
                           "StreamEvent(\(event.trackingId))",
                   categories: .streamSync)
        try cudaDevice.select()
        let event = event as! CudaStreamEvent
        try cudaCheck(status: cudaStreamWaitEvent(handle, event.handle, 0))
    }

    //--------------------------------------------------------------------------
    // waitUntilStreamIsComplete
    public func waitUntilStreamIsComplete() throws {
        assert(Thread.current === creatorThread, streamThreadViolationMessage)
        diagnostic("\(blockString) \(name) blocking caller until complete",
                   categories: .streamSync)
        try cudaCheck(status: cudaStreamSynchronize(handle))
    }

    //--------------------------------------------------------------------------
    /// simulateWork(x:timePerElement:result:
    /// introduces a delay in the stream by sleeping a duration of
    /// x.shape.elementCount * timePerElement
    public func simulateWork<T>(x: T, timePerElement: TimeInterval,
                                result: inout T)
            where T: TensorView
    {
        let delay = TimeInterval(x.shape.elementCount) * timePerElement
        delayStream(atLeast: delay)
    }

    //--------------------------------------------------------------------------
    /// delayStream(atLeast:
    /// causes the stream to sleep for the specified interval for testing
    public func delayStream(atLeast interval: TimeInterval) {
        assert(Thread.current === creatorThread, streamThreadViolationMessage)
//        queue {
//            Thread.sleep(forTimeInterval: interval)
//        }
    }

    //--------------------------------------------------------------------------
    /// throwTestError
    /// used for unit testing
    public func throwTestError() {
        assert(Thread.current === creatorThread, streamThreadViolationMessage)
//        queue {
//            throw DeviceError.streamError(idPath: [], message: "testError")
//        }
    }
} // CudaStream

////==============================================================================
//// cudaDataShape(from:)
//// DEFINED IN C Code
//public func cudaDataShape<T>(from tensor: T) -> cudaShape_t where
//    T: TensorView
//{
//    var ptr = UnsafeMutablePointer<cudaShape_t>.allocate(capacity: 1)
//    defer {
//        ptr.deinitialize(count: 1);
//        ptr.deallocate()
//    }
//
//    cudaInitCudaShape(
//            &ptr.pointee,
//            data.dataType.cuda,
//            data.shape.layout.cudnn,
//            data.extent.count,
//            data.extent,
//            data.strides,
//            data.shape.elementCount)
//
//    return ptr.pointee
//}

//==============================================================================
// CudaReductionContext
public final class CudaReductionContext: ReductionContext {
    // properties
    public let op: ReductionOp
    public let workspace: DeviceArray
    public let workspaceSizeInBytes: Int
    public let reduceTensorDesc: cudnnReduceTensorDescriptor_t
    public let inTensor: TensorDescriptor
    public let outTensor: TensorDescriptor

    //--------------------------------------------------------------------------
    // initializers
    public init(stream: CudaStream,
                op: ReductionOp,
                dataType: DataType,
                inTensor: TensorDescriptor,
                outTensor: TensorDescriptor) throws {

        self.op = op
        self.inTensor = inTensor
        self.outTensor = outTensor

        var temp: cudnnReduceTensorDescriptor_t?
        try cudaCheck(status: cudnnCreateReduceTensorDescriptor(&temp))
        reduceTensorDesc = temp!

        let indicesAction = (op == .min || op == .max) ?
                Cuda.CUDNN_REDUCE_TENSOR_FLATTENED_INDICES :
                Cuda.CUDNN_REDUCE_TENSOR_NO_INDICES

        // adjust intermediate data type if needed
        var reductionDataType: DataType
        switch dataType {
        case .real16F: reductionDataType = .real32F
        default: reductionDataType = dataType
        }

        try cudaCheck(status: cudnnSetReduceTensorDescriptor(
                reduceTensorDesc,
                op.cudnn,
                reductionDataType.cudnn,
                Cuda.CUDNN_PROPAGATE_NAN,
                indicesAction,
                Cuda.CUDNN_32BIT_INDICES
        ))

        // determine workspace size
        var tempWorkspaceSizeInBytes = 0
        try cudaCheck(status: cudnnGetReductionWorkspaceSize(
                stream.cudnn.handle,
                reduceTensorDesc,
                inTensor.desc,
                outTensor.desc,
                &tempWorkspaceSizeInBytes
        ))
        workspaceSizeInBytes = tempWorkspaceSizeInBytes
        workspace = try stream.device.createArray(count: workspaceSizeInBytes)
    }
}
