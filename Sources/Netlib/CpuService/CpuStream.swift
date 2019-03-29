//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import TensorFlow

public final class CpuStream : DeviceStream {
	//--------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public let device: ComputeDevice
    public let id: Int
	public let name: String
    public var logging: LogInfo?

    //--------------------------------------------------------------------------
    // initializers
    public init(logging: LogInfo,
                device: ComputeDevice,
                name: String, id: Int) throws {
        // init
        self.logging = logging
        self.device = device
        self.id = id
        self.name = name
        let path = logging.namePath
        trackingId = ObjectTracker.global.register(self, namePath: path)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
    
    //--------------------------------------------------------------------------
    // createEvent
    public func execute<T>(functionId: UUID, with parameters: T) throws {
        print("queueing id: \(functionId) with: \(parameters)")
    }
    
    //--------------------------------------------------------------------------
    // createEvent
    public func setup<T>(functionId: UUID, instanceId: UUID,
                         with parameters: T) throws {
        
    }
    
    //--------------------------------------------------------------------------
    // createEvent
    public func release(instanceId: UUID) throws {
        
    }

	//--------------------------------------------------------------------------
	// createEvent
	public func createEvent(options: StreamEventOptions) throws -> StreamEvent {
		return CpuStreamEvent(options: options)
	}

	public func debugDelay(seconds: Double) throws {
        
	}

	// blockCallerUntilComplete
	public func blockCallerUntilComplete() throws {
		
	}
	
	public func wait(for event: StreamEvent) throws {
		
	}

	// sync(with other
	public func sync(with other: DeviceStream, event: StreamEvent) throws {
	}

	public func record(event: StreamEvent) throws  -> StreamEvent {
		return event
	}

    //--------------------------------------------------------------------------
// intrinsic ops
    public func abs<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : SignedNumeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func add<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func all(x: TensorView<Bool>, result: inout TensorView<Bool>) throws {
        <#code#>
    }
    
    public func any(x: TensorView<Bool>, result: inout TensorView<Bool>) throws {
        <#code#>
    }
    
    public func approximatelyEqual<T>(lhs: TensorView<T>, rhs: TensorView<T>, tolerance: Double, result: inout TensorView<Bool>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    
    public func argmax<T>(x: TensorView<T>, squeezingAxis axis: Int, result: inout TensorView<Int32>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func argmin<T>(x: TensorView<T>, squeezingAxis axis: Int, result: inout TensorView<Int32>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func broadcast<T>(x: TensorView<T>, toShape shape: DataShape, result: inout TensorView<T>) throws where T : AnyScalar, T : TensorFlowScalar {
        <#code#>
    }
    
    public func cast<T, U>(from: TensorView<T>, to result: inout TensorView<U>) throws where T : AnyScalar, T : TensorFlowScalar, U : AnyScalar, U : TensorFlowScalar {
        <#code#>
    }
    
    public func ceil<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    
    public func concatenate<T>(view: TensorView<T>, with other: TensorView<T>, alongAxis axis: Int, result: inout TensorView<T>) throws where T : AnyScalar, T : TensorFlowScalar {
        <#code#>
    }
    
    public func cos<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    
    public func cohs<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    
    public func div<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func equal<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<Bool>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func exp<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    
    public func floor<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    
    public func greater<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<Bool>) throws where T : AnyScalar, T : Comparable, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func greaterOrEqual<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<Bool>) throws where T : AnyScalar, T : Comparable, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func less<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<Bool>) throws where T : AnyScalar, T : Comparable, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func lessOrEqual<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<Bool>) throws where T : AnyScalar, T : Comparable, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func log<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    
    public func logicalNot(x: TensorView<Bool>, result: inout TensorView<Bool>) throws {
        <#code#>
    }
    
    public func logicalAnd(lhs: TensorView<Bool>, rhs: TensorView<Bool>, result: inout TensorView<Bool>) throws {
        <#code#>
    }
    
    public func logicalOr(lhs: TensorView<Bool>, rhs: TensorView<Bool>, result: inout TensorView<Bool>) throws {
        <#code#>
    }
    
    public func logSoftmax<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    
    public func matmul<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func max<T>(x: TensorView<T>, squeezingAxes axes: [Int], result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func maximum<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func mean<T>(x: TensorView<T>, squeezingAxes axes: [Int], result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func min<T>(x: TensorView<T>, squeezingAxes axes: [Int], result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func minimum<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func mod<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func mul<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func neg(x: TensorView<Bool>, result: inout TensorView<Bool>) throws {
        <#code#>
    }
    
    public func notEqual<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<Bool>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func pad<T>(x: TensorView<T>, with margins: [(before: Int32, after: Int32)], fillValue: T, result: inout TensorView<T>) throws where T : AnyScalar, T : TensorFlowScalar {
        <#code#>
    }
    
    public func pow<T>(x: TensorView<T>, y: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func prod<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func rsqrt<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    
    public func replacing<T>(x: TensorView<T>, with other: TensorView<T>, where mask: TensorView<Bool>, result: inout TensorView<T>) throws where T : AnyScalar, T : TensorFlowScalar {
        <#code#>
    }
    
    public func sin<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    
    public func sinh<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    
    public func square<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func squaredDifference<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func sqrt<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    
    public func subtract<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func sum<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        <#code#>
    }
    
    public func tan<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    
    public func tanh<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        <#code#>
    }
    

}
