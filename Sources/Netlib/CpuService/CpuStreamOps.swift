//******************************************************************************
//  Created by Edward Connell on 3/29/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation
import TensorFlow


extension CpuStream {
    //--------------------------------------------------------------------------
    // intrinsic ops
    public func abs<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : SignedNumeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func add<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func all(x: TensorView<Bool>, result: inout TensorView<Bool>) throws {
        fatalError("Not implemented")
    }
    
    public func any(x: TensorView<Bool>, result: inout TensorView<Bool>) throws {
        fatalError("Not implemented")
    }
    
    public func approximatelyEqual<T>(lhs: TensorView<T>, rhs: TensorView<T>, tolerance: Double, result: inout TensorView<Bool>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func argmax<T>(x: TensorView<T>, squeezingAxis axis: Int, result: inout TensorView<Int32>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func argmin<T>(x: TensorView<T>, squeezingAxis axis: Int, result: inout TensorView<Int32>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func broadcast<T>(x: TensorView<T>, toShape shape: DataShape, result: inout TensorView<T>) throws where T : AnyScalar, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func cast<T, U>(from: TensorView<T>, to result: inout TensorView<U>) throws where T : AnyScalar, T : TensorFlowScalar, U : AnyScalar, U : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func ceil<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func concatenate<T>(view: TensorView<T>, with other: TensorView<T>, alongAxis axis: Int, result: inout TensorView<T>) throws where T : AnyScalar, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func cos<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func cohs<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func div<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func equal<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<Bool>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func exp<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func floor<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func greater<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<Bool>) throws where T : AnyScalar, T : Comparable, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func greaterOrEqual<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<Bool>) throws where T : AnyScalar, T : Comparable, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func less<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<Bool>) throws where T : AnyScalar, T : Comparable, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func lessOrEqual<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<Bool>) throws where T : AnyScalar, T : Comparable, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func log<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func logicalNot(x: TensorView<Bool>, result: inout TensorView<Bool>) throws {
        fatalError("Not implemented")
    }
    
    public func logicalAnd(lhs: TensorView<Bool>, rhs: TensorView<Bool>, result: inout TensorView<Bool>) throws {
        fatalError("Not implemented")
    }
    
    public func logicalOr(lhs: TensorView<Bool>, rhs: TensorView<Bool>, result: inout TensorView<Bool>) throws {
        fatalError("Not implemented")
    }
    
    public func logSoftmax<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func matmul<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func max<T>(x: TensorView<T>, squeezingAxes axes: [Int], result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func maximum<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func mean<T>(x: TensorView<T>, squeezingAxes axes: [Int], result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func min<T>(x: TensorView<T>, squeezingAxes axes: [Int], result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func minimum<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func mod<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func mul<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func neg(x: TensorView<Bool>, result: inout TensorView<Bool>) throws {
        fatalError("Not implemented")
    }
    
    public func notEqual<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<Bool>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    /// TODO: Probably don't need this. It can be accomplished via TensorView indexing
    public func pad<T>(x: TensorView<T>, with margins: [(before: Int32, after: Int32)], fillValue: T, result: inout TensorView<T>) throws where T : AnyScalar, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func pow<T>(x: TensorView<T>, y: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func prod<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func rsqrt<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func replacing<T>(x: TensorView<T>, with other: TensorView<T>, where mask: TensorView<Bool>, result: inout TensorView<T>) throws where T : AnyScalar, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func sin<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func sinh<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func square<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func squaredDifference<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func sqrt<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func subtract<T>(lhs: TensorView<T>, rhs: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func sum<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : Numeric, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func tan<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
    
    public func tanh<T>(x: TensorView<T>, result: inout TensorView<T>) throws where T : AnyScalar, T : FloatingPoint, T : TensorFlowScalar {
        fatalError("Not implemented")
    }
}
