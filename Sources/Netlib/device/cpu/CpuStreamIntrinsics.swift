//******************************************************************************
//  Created by Edward Connell on 4/16/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

public extension CpuStream {
    func abs<T>(x: T, result: inout T) where
        T: TensorView, T.Scalar: SignedNumeric, T.Scalar.Magnitude == T.Scalar
    {
        
    }
    
    //--------------------------------------------------------------------------
    /// add
    func add<T>(lhs: T, rhs: T, result: inout T) where
        T : TensorView, T.Scalar : Numeric
    {
        queue(#function, lhs, rhs, &result) { lhs, rhs, results in
            zip(lhs, rhs).map(to: &results) { $0 + $1 }
        }
    }
    
    //--------------------------------------------------------------------------
    /// all
    func all<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where
        T : TensorView, T.Scalar == Bool
    {
        queue(&result) { ref in
            let x = try x.values(using: self)
            let result = try ref.readWrite(using: self)
            
            for value in x where !value {
                result[0] = false
                return
            }
            result[0] = true
        }
    }
    
    //--------------------------------------------------------------------------
    /// any
    func any<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where
        T : TensorView, T.Scalar == Bool
    {
        queue(&result) { ref in
            let x = try x.values(using: self)
            let result = try ref.readWrite(using: self)
            
            for value in x where value {
                result[0] = true
                return
            }
            result[0] = false
        }
    }
    
    //--------------------------------------------------------------------------
    /// approximatelyEqual
    func approximatelyEqual<T>(lhs: T, rhs: T,
                               tolerance: T.Scalar,
                               result: inout T.BoolView) where
        T : TensorView, T.Scalar : AnyFloatingPoint,
        T.BoolView.Scalar == Bool
    {
        queue(&result) { ref in
            let lhs = try lhs.values(using: self)
            let rhs = try rhs.values(using: self)
            var results = try ref.mutableValues(using: self)
            
            zip(lhs, rhs).map(to: &results) { $0.0 - $0.1 <= tolerance }
        }
    }
    
    func argmax<T>(x: T, axes: Vector<IndexScalar>?, result: inout T.IndexView) where T : TensorView, T.Scalar : Numeric, T.IndexView.Scalar == IndexScalar {
        
    }
    
    func argmin<T>(x: T, axes: Vector<IndexScalar>?, result: inout T.IndexView) where T : TensorView, T.Scalar : Numeric, T.IndexView.Scalar == IndexScalar {
        
    }
    
    //--------------------------------------------------------------------------
    /// asum
    func asum<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where
        T: TensorView, T.Scalar: SignedNumeric, T.Scalar.Magnitude == T.Scalar
    {
        queue(&result) { ref in
            let x = try x.values(using: self)
            var results = try ref.mutableValues(using: self)
            
            x.reduce(to: &results, T.Scalar.zero) {
                $0 + $1.magnitude
            }
        }
    }
    
    func cast<T, R>(from: T, to result: inout R) where T : TensorView, R : TensorView, T.Scalar : AnyConvertable, R.Scalar : AnyConvertable {
        
    }
    
    func ceil<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func concatenate<T>(view: T, with other: T, alongAxis axis: Int, result: inout T) where T : TensorView {
        
    }
    
    func cos<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func cosh<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    //--------------------------------------------------------------------------
    /// div
    func div<T>(lhs: T, rhs: T, result: inout T) where
        T : TensorView, T.Scalar : FloatingPoint
    {
        queue(&result) { ref in
            let lhs = try lhs.values(using: self)
            let rhs = try rhs.values(using: self)
            var results = try ref.mutableValues(using: self)
            
            zip(lhs, rhs).map(to: &results) { $0 / $1 }
        }
    }
    
    //--------------------------------------------------------------------------
    /// equal
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Scalar: Equatable,
        T.BoolView.Scalar == Bool
    {
        queue(&result) { ref in
            let lhs = try lhs.values(using: self)
            let rhs = try rhs.values(using: self)
            var results = try ref.mutableValues(using: self)
            
            zip(lhs, rhs).map(to: &results) { $0 == $1 }
        }
    }
    
    //--------------------------------------------------------------------------
    /// exp
    func exp<T>(x: T, result: inout T) where
        T : TensorView, T.Scalar : FloatingPoint
    {
        
    }
    
    //--------------------------------------------------------------------------
    /// fill(result:with:
    func fill<T>(_ result: inout T, with value: T.Scalar) where T : TensorView {
        queue(&result) { ref in
            var result = try ref.mutableValues(using: self)
            
            for index in result.indices {
                result[index] = value
            }
        }
    }
    
    //--------------------------------------------------------------------------
    /// fillWithIndex(x:startAt:
    func fillWithIndex<T>(_ result: inout T, startAt: Int) where
        T : TensorView, T.Scalar: AnyNumeric
    {
        queue(&result) { ref in
            var value = startAt
            var result = try ref.mutableValues(using: self)

            for index in result.indices {
                result[index] = T.Scalar(any: value)
                value += 1
            }
        }
    }

    func floor<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func greater<T>(lhs: T, rhs: T, result: inout T.BoolView) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func greaterOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func less<T>(lhs: T, rhs: T, result: inout T.BoolView) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func lessOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where T : TensorView, T.Scalar : Numeric {
        
    }

    //--------------------------------------------------------------------------
    /// log(x:result:
    func log<T>(x: T, result: inout T) where
        T: TensorView, T.Scalar: AnyFloatingPoint
    {
        queue(&result) { ref in
            let x = try x.values(using: self)
            var results = try ref.mutableValues(using: self)
            
            x.map(to: &results) {
                T.Scalar(any: Foundation.log($0.asFloat))
            }
        }
    }
    
    func logicalNot<T>(x: T, result: inout T) where T : TensorView, T.Scalar == Bool {
        
    }
    
    func logicalAnd<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar == Bool {
        
    }
    
    func logicalOr<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar == Bool {
        
    }
    
    func logSoftmax<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func matmul<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func max<T>(x: T, squeezingAxes axes: [Int], result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func maximum<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func mean<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func min<T>(x: T, squeezingAxes axes: [Int], result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func minimum<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func mod<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func mul<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        queue(&result) { ref in
            let lhs = try lhs.values(using: self)
            let rhs = try rhs.values(using: self)
            var results = try ref.mutableValues(using: self)
            
            zip(lhs, rhs).map(to: &results) { $0 * $1 }
        }
    }
    
    func neg<T>(x: T, result: inout T) where T : TensorView, T.Scalar : SignedNumeric {
        
    }
    
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where T : TensorView, T.Scalar : Numeric {
        
    }

    //--------------------------------------------------------------------------
    // pow
    // TODO something is wrong, I shouldn't need to do this to interface
    // with math functions
    func pow<T>(x: T, y: T, result: inout T) where
        T : TensorView, T.Scalar : AnyNumeric
    {
        queue(&result) { ref in
            let x = try x.values(using: self)
            let y = try y.values(using: self)
            var results = try ref.mutableValues(using: self)
            
            zip(x, y).map(to: &results) {
                T.Scalar(any: Foundation.pow($0.asDouble, $1.asDouble))
            }
        }
    }

    //--------------------------------------------------------------------------
    // prod
    func prod<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where
        T : TensorView, T.Scalar : AnyNumeric
    {
//        if let axes = axes {
//            queue(&result) { ref in
//                let x = try x.values(using: self)
//                var results = try ref.mutableValues(using: self)
//
//                x.reduce(to: &results, T.Scalar(any: 1)) { $0 * $1 }
//            }
//        } else {
            queue(&result) { ref in
                let x = try x.values(using: self)
                var results = try ref.mutableValues(using: self)
                
                x.reduce(to: &results, T.Scalar(any: 1)) { $0 * $1 }
            }
//        }
    }
    
    func rsqrt<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func replacing<T>(x: T, with other: T, where mask: T.BoolView, result: inout T) where T : TensorView {
        
    }
    
    func sin<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func sinh<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func square<T>(x: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func squaredDifference<T>(lhs: T, rhs: T, result: inout T) where T : TensorView, T.Scalar : Numeric {
        
    }
    
    func sqrt<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    //--------------------------------------------------------------------------
    // subtract
    func subtract<T>(lhs: T, rhs: T, result: inout T) where
        T : TensorView, T.Scalar : Numeric
    {
        queue(&result) { ref in
            let lhs = try lhs.values(using: self)
            let rhs = try rhs.values(using: self)
            var results = try ref.mutableValues(using: self)
            
            zip(lhs, rhs).map(to: &results) { $0 - $1 }
        }
    }
    
    //--------------------------------------------------------------------------
    // sum
    func sum<T>(x: T, axes: Vector<IndexScalar>?, result: inout T) where
        T : TensorView, T.Scalar : Numeric
    {
        queue(&result) { ref in
            let x = try x.values(using: self)
            var results = try ref.mutableValues(using: self)
            
            x.reduce(to: &results, T.Scalar.zero) { $0 + $1 }
        }
    }
    
    func tan<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
    
    func tanh<T>(x: T, result: inout T) where T : TensorView, T.Scalar : FloatingPoint {
        
    }
}
