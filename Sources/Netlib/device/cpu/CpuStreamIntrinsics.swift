//******************************************************************************
//  Created by Edward Connell on 4/16/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

public extension CpuStream {
    func abs<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : SignedNumeric {
        
    }
    
    func add<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        var result = getViewReference(&result)
        queue {
            try zip(lhs.values(), rhs.values()).map(to: &result) { $0 + $1 }
        }
    }
    
    func all<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) throws where T : TensorView, T.Scalar == Bool {
        
    }
    
    func any<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) throws where T : TensorView, T.Scalar == Bool {
        
    }
    
    func approximatelyEqual<T>(lhs: T, rhs: T, tolerance: ScalarValue<T.Scalar>, result: inout T.BoolView) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func argmax<T>(x: T, axes: Vector<IndexScalar>?, result: inout T.IndexView) throws where T : TensorView, T.Scalar : Numeric, T.IndexView.Scalar == IndexScalar {
        
    }
    
    func argmin<T>(x: T, axes: Vector<IndexScalar>?, result: inout T.IndexView) throws where T : TensorView, T.Scalar : Numeric, T.IndexView.Scalar == IndexScalar {
        
    }
    
    func asum<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func cast<T, R>(from: T, to result: inout R) throws where T : TensorView, R : TensorView, T.Scalar : AnyConvertable, R.Scalar : AnyConvertable {
        
    }
    
    func ceil<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func concatenate<T>(view: T, with other: T, alongAxis axis: Int, result: inout T) throws where T : TensorView {
        
    }
    
    func cos<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func cosh<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func div<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) throws where T : TensorView {
        
    }
    
    func exp<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    //--------------------------------------------------------------------------
    /// fill(x:with:
    func fill<T>(x: T, with value: T.Scalar) throws where T : TensorView {
        commandQueue.async { [x] in
            var x = x
            var values = try! x.mutableDeviceValues(using: self)
            for i in values.startIndex..<values.endIndex {
                values[i] = value
            }
        }
    }
    
    //--------------------------------------------------------------------------
    /// fillWithIndex(x:startAt:
    func fillWithIndex<T>(x: T, startAt: Int) throws
        where T : TensorView, T.Scalar: AnyNumeric {
            commandQueue.async {
                var x = x
                var values = try! x.mutableDeviceValues(using: self)
                var value = startAt
                for i in values.startIndex..<values.endIndex {
                    values[i] = T.Scalar(any: value)
                    value += 1
                }
            }
    }

    func floor<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func greater<T>(lhs: T, rhs: T, result: inout T.BoolView) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func greaterOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func less<T>(lhs: T, rhs: T, result: inout T.BoolView) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func lessOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func log<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func logicalNot<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar == Bool {
        
    }
    
    func logicalAnd<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar == Bool {
        
    }
    
    func logicalOr<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar == Bool {
        
    }
    
    func logSoftmax<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func matmul<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func max<T>(x: T, squeezingAxes axes: [Int], result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func maximum<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func mean<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func min<T>(x: T, squeezingAxes axes: [Int], result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func minimum<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func mod<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func mul<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func neg<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : SignedNumeric {
        
    }
    
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func pow<T>(x: T, y: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func prod<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func rsqrt<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func replacing<T>(x: T, with other: T, where mask: T.BoolView, result: inout T) throws where T : TensorView {
        
    }
    
    func sin<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func sinh<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func square<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func squaredDifference<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func sqrt<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func subtract<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func sum<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func tan<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func tanh<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
}
