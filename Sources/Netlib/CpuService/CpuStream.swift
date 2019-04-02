//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

public final class CpuStream : DeviceStream {
    public func abs<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : SignedNumeric {
      
    }
    
    public func add<TL, TR>(lhs: TL, rhs: TR, result: inout TL) throws where TL : TensorView, TR : TensorView, TL.Scalar : Numeric, TR.Scalar : Numeric {
      
    }
    
    public func all<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar == Bool {
      
    }
    
    public func any<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar == Bool {
      
    }
    
    public func approximatelyEqual<T, R>(lhs: T, rhs: T, tolerance: Double, result: inout R) throws where T : TensorView, R : TensorView, T.Scalar : FloatingPoint, R.Scalar == Bool {
      
    }
    
    public func argmax<T, R>(x: T, squeezingAxis axis: Int, result: inout R) throws where T : TensorView, R : TensorView, T.Scalar : Numeric, R.Scalar == Int32 {
      
    }
    
    public func argmin<T, R>(x: T, squeezingAxis axis: Int, result: inout R) throws where T : TensorView, R : TensorView, T.Scalar : Numeric, R.Scalar == Int32 {
      
    }
    
    public func broadcast<T>(x: T, toShape shape: DataShape, result: inout T) throws where T : TensorView {
      
    }
    
    public func cast<T, R>(from: T, to result: inout R) throws where T : TensorView, R : TensorView, T.Scalar : AnyNumeric, R.Scalar : AnyNumeric {
      
    }
    
    public func ceil<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
      
    }
    
    public func concatenate<T>(view: T, with other: T, alongAxis axis: Int, result: inout T) throws where T : TensorView {
      
    }
    
    public func cos<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
      
    }
    
    public func cosh<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
      
    }
    
    public func div<TL, TR>(lhs: TL, rhs: TR, result: inout TL) throws where TL : TensorView, TR : TensorView, TL.Scalar : Numeric, TR.Scalar : Numeric {
      
    }
    
    public func equal<T, R>(lhs: T, rhs: T, result: inout R) throws where T : TensorView, R : TensorView, T.Scalar : Numeric, R.Scalar == Bool {
      
    }
    
    public func exp<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
      
    }
    
    public func floor<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
      
    }
    
    public func greater<T, R>(lhs: T, rhs: T, result: inout R) throws where T : TensorView, R : TensorView, T.Scalar : Comparable, T.Scalar : Numeric, R.Scalar == Bool {
      
    }
    
    public func greaterOrEqual<T, R>(lhs: T, rhs: T, result: inout R) throws where T : TensorView, R : TensorView, T.Scalar : Comparable, T.Scalar : Numeric, R.Scalar == Bool {
      
    }
    
    public func less<T, R>(lhs: T, rhs: T, result: inout R) throws where T : TensorView, R : TensorView, T.Scalar : Comparable, T.Scalar : Numeric, R.Scalar == Bool {
      
    }
    
    public func lessOrEqual<T, R>(lhs: T, rhs: T, result: inout R) throws where T : TensorView, R : TensorView, T.Scalar : Comparable, T.Scalar : Numeric, R.Scalar == Bool {
      
    }
    
    public func log<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
      
    }
    
    public func logicalNot<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar == Bool {
      
    }
    
    public func logicalAnd<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar == Bool {
      
    }
    
    public func logicalOr<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar == Bool {
      
    }
    
    public func logSoftmax<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
      
    }
    
    public func matmul<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
      
    }
    
    public func max<T>(x: T, squeezingAxes axes: [Int], result: inout T) throws where T : TensorView, T.Scalar : Numeric {
      
    }
    
    public func maximum<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
      
    }
    
    public func mean<T>(x: T, squeezingAxes axes: [Int], result: inout T) throws where T : TensorView, T.Scalar : Numeric {
      
    }
    
    public func min<T>(x: T, squeezingAxes axes: [Int], result: inout T) throws where T : TensorView, T.Scalar : Numeric {
      
    }
    
    public func minimum<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
      
    }
    
    public func mod<TL, TR>(lhs: TL, rhs: TR, result: inout TL) throws where TL : TensorView, TR : TensorView, TL.Scalar : Numeric, TR.Scalar : Numeric {
      
    }
    
    public func mul<TL, TR>(lhs: TL, rhs: TR, result: inout TL) throws where TL : TensorView, TR : TensorView, TL.Scalar : Numeric, TR.Scalar : Numeric {
      
    }
    
    public func neg<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : SignedNumeric {
      
    }
    
    public func notEqual<TL, TR, R>(lhs: TL, rhs: TR, result: inout R) throws where TL : TensorView, TR : TensorView, R : TensorView, TL.Scalar : Comparable, TL.Scalar : Numeric, TR.Scalar : Comparable, TR.Scalar : Numeric, R.Scalar == Bool {
      
    }
    
    public func pad<T>(x: T, with margins: [(before: Int, after: Int)], fillValue: T.Scalar, result: inout T) throws where T : TensorView {
      
    }
    
    public func pow<TX, TY, TR>(x: TX, y: TY, result: inout TR) throws where TX : TensorView, TY : TensorView, TR : TensorView, TX.Scalar : Numeric, TY.Scalar : Numeric, TR.Scalar : Numeric {
      
    }
    
    public func prod<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
      
    }
    
    public func rsqrt<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
      
    }
    
    public func replacing<T, R>(x: T, with other: T, where mask: R, result: inout T) throws where T : TensorView, R : TensorView, R.Scalar == Bool {
      
    }
    
    public func sin<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
      
    }
    
    public func sinh<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
      
    }
    
    public func square<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
      
    }
    
    public func squaredDifference<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
      
    }
    
    public func sqrt<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
      
    }
    
    public func subtract<TL, TR>(lhs: TL, rhs: TR, result: inout TL) throws where TL : TensorView, TR : TensorView, TL.Scalar : Numeric, TR.Scalar : Numeric {
      
    }
    
    public func sum<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
      
    }
    
    public func tan<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
      
    }
    
    public func tanh<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
      
    }
    
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
}
