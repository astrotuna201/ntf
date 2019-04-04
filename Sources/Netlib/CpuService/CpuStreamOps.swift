//******************************************************************************
//  Created by Edward Connell on 3/29/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation
import TensorFlow


extension CpuStream {
    public func abs<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : SignedNumeric {
        
    }
    
    public func add<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        let _ = try lhs.readOnly()[0] + rhs.readOnly()[0]
    }
    
    public func all<T>(x: T, axes: VectorTensor<TensorIndex>?, result: inout T) throws where T : TensorView, T.Scalar == Bool {
        
    }
    
    public func any<T>(x: T, axes: VectorTensor<TensorIndex>?, result: inout T) throws where T : TensorView, T.Scalar == Bool {
        
    }
    
    public func approximatelyEqual<T>(lhs: T, rhs: T, tolerance: ScalarTensor<T.Scalar>, result: inout T.BoolView) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    public func argmax<T>(x: T, axes: VectorTensor<TensorIndex>?, result: inout T.IndexView) throws where T : TensorView, T.Scalar : Numeric, T.IndexView.Scalar == TensorIndex {
        
    }
    
    public func argmin<T>(x: T, axes: VectorTensor<TensorIndex>?, result: inout T.IndexView) throws where T : TensorView, T.Scalar : Numeric, T.IndexView.Scalar == TensorIndex {
        
    }
    
    public func broadcast<T>(x: T, toShape shape: DataShape, result: inout T) throws where T : TensorView {
        
    }
    
    public func cast<T, R>(from: T, to result: inout R) throws where T : TensorView, R : TensorView, T.Scalar : AnyConvertable, R.Scalar : AnyConvertable {
        
    }
    
    public func ceil<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    public func concatenate<T>(view: T, with other: T, alongAxis axis: Int, result: inout T) throws where T : TensorView {
        
    }
    
    public func cos<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    public func cosh<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    public func div<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) throws where T : TensorView {
        
    }
    
    public func exp<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    public func floor<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    public func greater<T>(lhs: T, rhs: T, result: inout T.BoolView) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func greaterOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func less<T>(lhs: T, rhs: T, result: inout T.BoolView) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func lessOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) throws where T : TensorView, T.Scalar : Numeric {
        
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
    
    public func mean<T>(x: T, axes: VectorTensor<TensorIndex>?, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func min<T>(x: T, squeezingAxes axes: [Int], result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func minimum<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func mod<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func mul<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func neg<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : SignedNumeric {
        
    }
    
    public func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func pow<T>(x: T, y: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func prod<T>(x: T, axes: VectorTensor<TensorIndex>?, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func rsqrt<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    public func replacing<T>(x: T, with other: T, where mask: T.BoolView, result: inout T) throws where T : TensorView {
        
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
    
    public func subtract<T>(lhs: T, rhs: T, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func sum<T>(x: T, axes: VectorTensor<TensorIndex>?, result: inout T) throws where T : TensorView, T.Scalar : Numeric {
        
    }
    
    public func tan<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    public func tanh<T>(x: T, result: inout T) throws where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    
}
