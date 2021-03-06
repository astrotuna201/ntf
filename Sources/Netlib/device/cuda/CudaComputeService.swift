//******************************************************************************
//  Created by Edward Connell on 4/4/16
//  Copyright © 2016 Connell Research. All rights reserved.
//

import Foundation
import Cuda

//==============================================================================
// CudaComputeService
public final class CudaComputeService: LocalComputeService {
    // properties
    public private(set) weak var platform: ComputePlatform!
    public private(set) var trackingId = 0
    public private(set) var devices = [ComputeDevice]()
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error?
    public var _errorMutex: Mutex = Mutex()
    public let id: Int
    public var logInfo: LogInfo
    public let name: String

    //--------------------------------------------------------------------------
    // timeout
    public var timeout: TimeInterval? {
        didSet {
            devices.forEach {
                $0.timeout = timeout
            }
        }
    }

    //--------------------------------------------------------------------------
    // initializers
    public required init(platform: ComputePlatform,
                         id: Int,
                         logInfo: LogInfo,
                         name: String? = nil) throws {
        self.platform = platform
        self.id = id
        self.name = name ?? "cuda"
        self.logInfo = logInfo

        // this is held statically by the Platform
        trackingId = ObjectTracker.global.register(self, isStatic: true)

        // create devices
        var deviceCount: CInt = 0
        do {
            try cudaCheck(status: cudaGetDeviceCount(&deviceCount))
        } catch {
            writeLog("cudaGetDeviceCount failed. " +
                             "The Cuda driver may be in an unstable state",
                     level: .error)
            throw error
        }

        guard deviceCount > 0 else {
            writeLog("There are no '\(self.name)' devices installed", level: .warning)
            throw ServiceError.serviceIsUnavailable
        }

        // add device object for each id reported
        for i in 0..<Int(deviceCount) {
            let device = try CudaDevice(service: self, deviceId: i,
                                        logInfo: logInfo.flat("gpu:\(i)"),
                                        memoryAddressing: .discreet,
                                        timeout: timeout)
            devices.append(device)
        }
    }

    deinit {
        ObjectTracker.global.remove(trackingId: trackingId)
    }
}

//==============================================================================
// cudaCheck cudaError_t
public func cudaCheck(status: cudaError_t, file: String = #file,
                      function: String = #function, line: Int = #line) throws {
    if status != cudaSuccess {
        let location = "CUDA error in \(file) at \(function):\(line)"
        let message = String(utf8String: cudaGetErrorString(status))!
        cudaDeviceReset()
        throw ServiceError.functionFailure(location: location, message: message)
    }
}

//==============================================================================
// cudaCheck cudnnStatus_t
public func cudaCheck(status: cudnnStatus_t, file: String = #file,
                      function: String = #function, line: Int = #line) throws {
    if status != CUDNN_STATUS_SUCCESS {
        let location = "CUDNN error in \(file) at \(function):\(line)"
        let message = String(utf8String: cudnnGetErrorString(status))!
        print(message)
        cudaDeviceReset()
        throw ServiceError.functionFailure(location: location, message: message)
    }
}

//==============================================================================
// cudaCheck cublasStatus_t
public func cudaCheck(status: cublasStatus_t, file: String = #file,
                      function: String = #function, line: Int = #line) throws {
    if status != CUBLAS_STATUS_SUCCESS {
        let location = "CUBLAS error in \(file) at \(function):\(line)"
        let message = String(utf8String: cublasGetErrorString(status))! + "code=(\(status))"
        cudaDeviceReset()
        throw ServiceError.functionFailure(location: location, message: message)
    }
}

public func cublasGetErrorString(_ status: cublasStatus_t) -> String {
    switch status {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS"
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED"
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED"
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR"
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED"
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR"
    default: return "<unknown>"
    }
}

//==============================================================================
// cudaCheck curandStatus_t
public func cudaCheck(status: curandStatus_t, file: String = #file,
                      function: String = #function, line: Int = #line) throws {
    if status != CURAND_STATUS_SUCCESS {
        let location = "CURAND error in \(file) at \(function):\(line)"
        let message = String(utf8String: curandGetErrorString(status))! + "code=(\(status))"
        cudaDeviceReset()
        throw ServiceError.functionFailure(location: location, message: message)
    }
}

public func curandGetErrorString(_ status: curandStatus_t) -> String {
    switch status {
    case CURAND_STATUS_SUCCESS:    return "CURAND_STATUS_SUCCESS"
    case CURAND_STATUS_VERSION_MISMATCH:    return "CURAND_STATUS_VERSION_MISMATCH"
    case CURAND_STATUS_NOT_INITIALIZED:    return "CURAND_STATUS_NOT_INITIALIZED"
    case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED"
    case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR"
    case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE"
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE"
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED"
    case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE"
    case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE"
    case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED"
    case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH"
    case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR"
    default: return "<unknown>"
    }
}

//==============================================================================
/// ReductionOp
public enum ReductionOp {
    case add, mul, min, max, amax, avg, norm1, norm2
}

//==============================================================================
/// NanPropagation
public enum NanPropagation {
    case propagate, noPropagate
}

//==============================================================================
/// ReductionContext
public protocol ReductionContext {
}

////------------------------------------------------------------------------------
//// TransposeOp
//extension TransposeOp {
//	public var cublas: cublasOperation_t {
//		switch self {
//		case .noTranspose: return CUBLAS_OP_N
//		case .transpose: return CUBLAS_OP_T
//		case .conjugateTranspose: return CUBLAS_OP_C
//		}
//	}
//}

//------------------------------------------------------------------------------
// ReductionOp extension
extension ReductionOp {
    public var cudnn: cudnnReduceTensorOp_t {
        get {
            switch self {
            case .add: return CUDNN_REDUCE_TENSOR_ADD
            case .mul: return CUDNN_REDUCE_TENSOR_MUL
            case .min: return CUDNN_REDUCE_TENSOR_MIN
            case .max: return CUDNN_REDUCE_TENSOR_MAX
            case .amax: return CUDNN_REDUCE_TENSOR_AMAX
            case .avg: return CUDNN_REDUCE_TENSOR_AVG
            case .norm1: return CUDNN_REDUCE_TENSOR_NORM1
            case .norm2: return CUDNN_REDUCE_TENSOR_NORM2
            }
        }
    }
}

//------------------------------------------------------------------------------
// DataType extension
extension DataType {
    public init(cudnn: cudnnDataType_t) {
        switch cudnn {
        case CUDNN_DATA_INT8:   self = .real8U
        case CUDNN_DATA_INT32:  self = .real32I
        case CUDNN_DATA_HALF:   self = .real16F
        case CUDNN_DATA_FLOAT:  self = .real32F
        case CUDNN_DATA_DOUBLE: self = .real64F
        default: fatalError("Invalid state")
        }
    }

    public var cudnn: cudnnDataType_t {
        get {
            switch self {
            case .real8U: return CUDNN_DATA_INT8
            case .real32I: return CUDNN_DATA_INT32
            case .real16F: return CUDNN_DATA_HALF
            case .real32F: return CUDNN_DATA_FLOAT
            case .real64F: return CUDNN_DATA_DOUBLE
            default: fatalError("Invalid state")
            }
        }
    }

    public var cuda: cudaDataType {
        get {
            switch self {
            case .real16F: return CUDA_R_16F
            case .real32F: return CUDA_R_32F
            case .real64F: return CUDA_R_64F
            case .real8U: return CUDA_R_8U
            case .real32I: return CUDA_R_32I
            default: fatalError("not supported")
            }
        }
    }
}

//------------------------------------------------------------------------------
// NanPropagation
extension NanPropagation {
    public var cudnn: cudnnNanPropagation_t {
        get {
            switch self {
            case .noPropagate: return CUDNN_NOT_PROPAGATE_NAN
            case .propagate: return CUDNN_PROPAGATE_NAN
            }
        }
    }
}

//==============================================================================
// CudnnHandle
public final class CudnnHandle: ObjectTracking {
    init(deviceId: Int, using stream: cudaStream_t, isStatic: Bool) throws {
        self.deviceId = deviceId
        try cudaCheck(status: cudaSetDevice(Int32(deviceId)))

        var temp: cudnnHandle_t?
        try cudaCheck(status: cudnnCreate(&temp))
        handle = temp!
        try cudaCheck(status: cudnnSetStream(handle, stream))
        trackingId = ObjectTracker.global.register(self, isStatic: isStatic)
    }

    deinit {
        do {
            try cudaCheck(status: cudaSetDevice(Int32(deviceId)))
            try cudaCheck(status: cudnnDestroy(handle))
            ObjectTracker.global.remove(trackingId: trackingId)
        } catch {
            print("\(releaseString) CudnnHandle(\(trackingId)) \(String(describing: error))")
        }
    }

    // properties
    public private (set) var trackingId = 0
    private let deviceId: Int
    public var handle: cudnnHandle_t
}

//==============================================================================
// CublasHandle
public final class CublasHandle: ObjectTracking {
    public init(deviceId: Int, using stream: cudaStream_t,
                isStatic: Bool) throws
    {
        self.deviceId = deviceId
        try cudaCheck(status: cudaSetDevice(Int32(deviceId)))

        var temp: cublasHandle_t?
        try cudaCheck(status: cublasCreate_v2(&temp))
        handle = temp!
        try cudaCheck(status: cublasSetStream_v2(handle, stream))
        trackingId = ObjectTracker.global.register(self, isStatic: isStatic)
    }

    deinit {
        do {
            try cudaCheck(status: cudaSetDevice(Int32(deviceId)))
            try cudaCheck(status: cublasDestroy_v2(handle))
            ObjectTracker.global.remove(trackingId: trackingId)
        } catch {
            print(String(describing: error))
        }
    }

    // properties
    public private (set) var trackingId = 0
    private let deviceId: Int
    public var handle: cublasHandle_t
}

////==============================================================================
//// ActivationDescriptor
//public final class ActivationDescriptor : ObjectTracking {
//	public init(mode: ActivationMode, nan: NanPropagation, reluCeiling: Double) throws {
//		// create the descriptor
//		var temp: cudnnActivationDescriptor_t?
//		try cudaCheck(status: cudnnCreateActivationDescriptor(&temp))
//		desc = temp!
//
//		// initialize
//		try cudaCheck(status: cudnnSetActivationDescriptor(
//			desc, mode.cudnn, nan.cudnn, reluCeiling))
//		trackingId = ObjectTracker.global.register(self)
//	}
//
//	deinit {
//		try! cudaCheck(status: cudnnDestroyActivationDescriptor(desc))
//		ObjectTracker.global.remove(trackingId: trackingId)
//	}
//
//	// properties
//	public private (set) var trackingId = 0
//	let desc: cudnnActivationDescriptor_t
//}
//
////--------------------------------------
//// ActivationMode
//public enum ActivationMode {
//    case sigmoid, relu, tanh, clippedRelu
//}
//
////==============================================================================
//// ConvolutionDescriptor
//public final class ConvolutionDescriptor : ObjectTracking {
//	// initializers
//	public init(dataType: DataType, rank: Int, pad: [Int],
//	            stride: [Int], dilation: [Int], mode: ConvolutionMode) throws {
//		// create the descriptor
//		var temp: cudnnConvolutionDescriptor_t?
//		try cudaCheck(status: cudnnCreateConvolutionDescriptor(&temp))
//		desc = temp!
//
//		// initialize
//		try cudaCheck(status: cudnnSetConvolutionNdDescriptor(
//			desc, CInt(rank),
//			pad.map { CInt($0) },
//			stride.map { CInt($0) },
//			dilation.map { CInt($0) },
//			mode.cudnn,
//			dataType.cudnn))
//
//		trackingId = ObjectTracker.global.register(self)
//	}
//
//	deinit {
//		try! cudaCheck(status: cudnnDestroyConvolutionDescriptor(desc))
//		ObjectTracker.global.remove(trackingId: trackingId)
//	}
//
//	// properties
//	public private (set) var trackingId = 0
//	let desc: cudnnConvolutionDescriptor_t
//}
//
//// ConvolutionMode
//public enum ConvolutionMode {
//    case convolution
//    case crossCorrelation
//}
//
//////==============================================================================
////// DropoutDescriptor
////public final class DropoutDescriptor : ObjectTracking {
////	// initializers
////	public init(stream: CudaStream, drop: Double, seed: UInt64,
////	            tensorDesc: TensorDescriptor) throws {
////		// create the descriptor
////		var temp: cudnnDropoutDescriptor_t?
////		try cudaCheck(status: cudnnCreateDropoutDescriptor(&temp))
////		desc = temp!
////
////		// get states size
////		var stateSizeInBytes = 0
////		try cudaCheck(status: cudnnDropoutGetStatesSize(
////			tensorDesc.desc, &stateSizeInBytes))
////
////		// create states array
////		states = try stream.device.createArray(count: stateSizeInBytes)
////
////		// initialize
////		try cudaCheck(status: cudnnSetDropoutDescriptor(
////			desc,
////			stream.cudnn.handle,
////			Float(drop),
////			states.data,
////			states.count,
////			seed
////		))
////
////		trackingId = ObjectTracker.global.register(self)
////	}
////
////	deinit {
////		try! cudaCheck(status: cudnnDestroyDropoutDescriptor(desc))
////		ObjectTracker.global.remove(trackingId: trackingId)
////	}
////
////	// properties
////	private var states: DeviceArray
////	public private (set) var trackingId = 0
////	let desc: cudnnDropoutDescriptor_t
////}
//
////==============================================================================
//// FilterDescriptor
//public final class FilterDescriptor : ObjectTracking {
//	// initializers
//	public init(shape: DataShape, dataType: DataType) throws {
//		// create the descriptor
//		var temp: cudnnFilterDescriptor_t?
//		try cudaCheck(status: cudnnCreateFilterDescriptor(&temp))
//		desc = temp!
//
//		// initialize
//		try cudaCheck(status: cudnnSetFilterNdDescriptor(
//			desc, dataType.cudnn,
//			shape.layout.cudnn,
//			Int32(shape.extents.count),
//			shape.extents.map { Int32($0)}))
//
//		trackingId = ObjectTracker.global.register(self)
//	}
//
//	deinit {
//		try! cudaCheck(status: cudnnDestroyFilterDescriptor(desc))
//		ObjectTracker.global.remove(trackingId: trackingId)
//	}
//
//	// properties
//	public private (set) var trackingId = 0
//	let desc: cudnnFilterDescriptor_t
//}
//
////==============================================================================
//// LRNDescriptor
//public final class LRNDescriptor: ObjectTracking {
//	// initializers
//	public init(N: Int, alpha: Double, beta: Double, K: Double) throws {
//		guard N >= Int(CUDNN_LRN_MIN_N) && N <= Int(CUDNN_LRN_MAX_N) else {
//			throw ServiceError.rangeError(
//				"N = \(N) is invalid. Range \(CUDNN_LRN_MIN_N) " +
//                        "to \(CUDNN_LRN_MAX_N)")
//		}
//		guard K >= CUDNN_LRN_MIN_K else {
//			throw ServiceError.rangeError(
//				"K = \(K) is invalid. Must be >= to \(CUDNN_LRN_MIN_K)")
//		}
//		guard beta >= CUDNN_LRN_MIN_BETA else {
//			throw ServiceError.rangeError(
//				"beta = \(beta) is invalid. Must be >= to \(CUDNN_LRN_MIN_BETA)")
//		}
//
//		// create the descriptor
//		var temp: cudnnLRNDescriptor_t?
//		try cudaCheck(status: cudnnCreateLRNDescriptor(&temp))
//		desc = temp!
//
//		// initialize
//		try cudaCheck(status: cudnnSetLRNDescriptor(desc, CUnsignedInt(N), alpha, beta, K))
//		trackingId = ObjectTracker.global.register(self)
//	}
//
//	deinit {
//		try! cudaCheck(status: cudnnDestroyLRNDescriptor(desc))
//		ObjectTracker.global.remove(trackingId: trackingId)
//	}
//
//	// properties
//	public private (set) var trackingId = 0
//	let desc: cudnnLRNDescriptor_t
//}
//
//==============================================================================
// TensorDescriptor
public final class TensorDescriptor: ObjectTracking {
    // initializers
    public init(shape: DataShape, dataType: DataType) throws {
        // create the descriptor
        var temp: cudnnTensorDescriptor_t?
        try cudaCheck(status: cudnnCreateTensorDescriptor(&temp))
        desc = temp!

        // initialize
        try cudaCheck(status: cudnnSetTensorNdDescriptor(
                desc, dataType.cudnn,
                Int32(shape.extents.count),
                shape.extents.map {
                    Int32($0)
                },
                shape.strides.map {
                    Int32($0)
                }))

        trackingId = ObjectTracker.global.register(self)
    }

    public init(owning desc: cudnnTensorDescriptor_t) {
        self.desc = desc
        trackingId = ObjectTracker.global.register(self)
    }

    deinit {
        try! cudaCheck(status: cudnnDestroyTensorDescriptor(desc))
        ObjectTracker.global.remove(trackingId: trackingId)
    }

    // properties
    public private (set) var trackingId = 0
    let desc: cudnnTensorDescriptor_t

    // getInfo
    public func getInfo() throws -> (extent: [Int], strides: [Int], DataType) {
        let reqDims = Int(CUDNN_DIM_MAX)
        var dims = [Int32](repeating: 0, count: reqDims)
        var strides = [Int32](repeating: 0, count: reqDims)
        var type = cudnnDataType_t(0)
        var numDims: Int32 = 0

        try cudaCheck(status: cudnnGetTensorNdDescriptor(
                desc,
                Int32(reqDims),
                &type,
                &numDims,
                &dims,
                &strides
        ))

        return (
                dims[0..<Int(numDims)].map {
                    Int($0)
                },
                strides[0..<Int(numDims)].map {
                    Int($0)
                },
                DataType(cudnn: type))
    }
}

////==============================================================================
//// createTensorDescriptor
//extension TensorView {
//	public func createTensorDescriptor(asShape newShape: DataShape? = nil)
//    throws -> TensorDescriptor
//    {
//		assert(newShape == nil || newShape!.elementCount == shape.elementCount)
//		return try TensorDescriptor(shape: newShape ?? shape, dataType: dataType)
//	}
//}
//
////==============================================================================
//// PoolingDescriptor
//public final class PoolingDescriptor : ObjectTracking {
//	// initializers
//	public init(mode: PoolingMode, nan: NanPropagation, rank: Int, window: [Int],
//	            padding: [Int], stride: [Int]) throws {
//		// create the descriptor
//		var temp: cudnnPoolingDescriptor_t?
//		try cudaCheck(status: cudnnCreatePoolingDescriptor(&temp))
//		desc = temp!
//
//		// initialize
//		try cudaCheck(status: cudnnSetPoolingNdDescriptor(
//			desc, mode.cudnn, nan.cudnn,
//			CInt(rank),
//			window.map { CInt($0) },
//			padding.map { CInt($0) },
//			stride.map { CInt($0) }))
//
//		trackingId = ObjectTracker.global.register(self)
//	}
//
//	deinit {
//		try! cudaCheck(status: cudnnDestroyLRNDescriptor(desc))
//		ObjectTracker.global.remove(trackingId: trackingId)
//	}
//
//	// properties
//	public private (set) var trackingId = 0
//	let desc: cudnnPoolingDescriptor_t
//}
//
////--------------------------------------
//// PoolingMode
//public enum PoolingMode {
//    case averageExcludePadding, averageIncludePadding, max
//}
//
////==============================================================================
//// ReduceTensorDescriptor
//public final class ReduceTensorDescriptor : ObjectTracking {
//    // properties
//    public private (set) var trackingId = 0
//    let desc: cudnnReduceTensorDescriptor_t
//
//	// initializers
//	public init(op: ReductionOp, nan: NanPropagation, dataType: DataType) throws {
//		// create the descriptor
//		var temp: cudnnReduceTensorDescriptor_t?
//		try cudaCheck(status: cudnnCreateReduceTensorDescriptor(&temp))
//		desc = temp!
//
//		let indicesAction = (op == .min || op == .max) ?
//			CUDNN_REDUCE_TENSOR_FLATTENED_INDICES : CUDNN_REDUCE_TENSOR_NO_INDICES
//
//		// initialize
//		try cudaCheck(status: cudnnSetReduceTensorDescriptor(
//			desc,
//			op.cudnn,
//			dataType == .real64F ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT,
//			nan.cudnn,
//			indicesAction,
//			CUDNN_32BIT_INDICES
//		))
//
//		trackingId = ObjectTracker.global.register(self)
//	}
//
//	deinit {
//		try! cudaCheck(status: cudnnDestroyReduceTensorDescriptor(desc))
//        ObjectTracker.global.remove(trackingId: trackingId)
//	}
//}
