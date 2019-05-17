//******************************************************************************
//  Created by Edward Connell on 4/16/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

public extension CpuStream {
    //--------------------------------------------------------------------------
    /// abs
    func abs<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: FloatingPoint
    {
        queue(#function, { try x.values() }, &result) {
            $0.map(to: &$1) { $0.magnitude }
        }
    }

    //--------------------------------------------------------------------------
    /// add
    func add<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Values.Element: Numeric
    {
        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
            zip($0.0, $0.1).map(to: &$1) { $0 + $1 }
        }
    }
    
    //--------------------------------------------------------------------------
    /// all
    func all<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Values.Element == Bool
    {
        queue(#function, { try x.values() }, &result) {
            $1[$1.startIndex] = $0.first { $0 == false } != nil
        }
    }
    
    //--------------------------------------------------------------------------
    /// any
    func any<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Values.Element == Bool
    {
        queue(#function, { try x.values() }, &result) {
            $1[$1.startIndex] = $0.first { $0 == true } != nil
        }
    }
    
    //--------------------------------------------------------------------------
    /// approximatelyEqual
    func approximatelyEqual<T>(lhs: T, rhs: T,
                               tolerance: T.Values.Element,
                               result: inout T.BoolView) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
            zip($0.0, $0.1).map(to: &$1) { $0.0 - $0.1 <= tolerance }
        }
    }
    
    //--------------------------------------------------------------------------
    /// argmax
    func argmax<T>(x: T, along axes: Vector<IndexElement>?,
                   result: inout T.IndexView) where
        T: TensorView, T.Values.Element: Numeric
    {
        
    }
    
    //--------------------------------------------------------------------------
    /// argmin
    func argmin<T>(x: T, along axes: Vector<IndexElement>?,
                   result: inout T.IndexView) where
        T: TensorView, T.Values.Element: Numeric
    {
        
    }
    
    //--------------------------------------------------------------------------
    /// asum
    func asum<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Values.Element: FloatingPoint
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.reduce(to: &$1, T.Values.Element.zero) {
//                $0 + $1.magnitude
//            }
//        }
    }
    
    //--------------------------------------------------------------------------
    /// cast
    func cast<T, R>(from: T, to result: inout R) where
        T: TensorView, R: TensorView, T.Values.Element: AnyConvertable,
        R.Element : AnyConvertable
    {
        
    }
    
    //--------------------------------------------------------------------------
    /// ceil
    func ceil<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.map(to: &$1) {
//                T.Values.Element(any: Foundation.ceilf($0.asFloat))
//            }
//        }
    }
    
    //--------------------------------------------------------------------------
    /// concatenate
    func concatenate<T>(view: T, with other: T,
                        alongAxis axis: Int, result: inout T) where
        T: TensorView
    {
        
    }
    
    //--------------------------------------------------------------------------
    /// copies the elements from view to result
    func copy<T, R>(view: T, result: inout R) where
        T: TensorView, R: TensorView,
        R.MutableValues.Element == T.Values.Element
    {
//        queue(#function, { try view.values() }, &result) {
//            $0.map(to: &$1) { $0 }
//        }
    }
    
    //--------------------------------------------------------------------------
    /// cos
    func cos<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.map(to: &$1) {
//                T.Values.Element(any: Foundation.cos($0.asFloat))
//            }
//        }
    }
    
    //--------------------------------------------------------------------------
    /// cosh
    func cosh<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.map(to: &$1) {
//                T.Values.Element(any: Foundation.cosh($0.asFloat))
//            }
//        }
    }
    
    //--------------------------------------------------------------------------
    /// div
    func div<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Values.Element: FloatingPoint
    {
        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
            zip($0.0, $0.1).map(to: &$1) { $0 / $1 }
        }
    }
    
    //--------------------------------------------------------------------------
    /// equal
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Values.Element: Equatable
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) { $0 == $1 }
//        }
    }
    
    //--------------------------------------------------------------------------
    /// exp
    func exp<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.map(to: &$1) {
//                T.Values.Element(any: Foundation.log($0.asFloat))
//            }
//        }
    }
    
    //--------------------------------------------------------------------------
    /// fill(result:with:
    /// NOTE: this can be much faster, doesn't need to be ordered access
    func fill<T>(_ result: inout T, with value: T.Values.Element) where T: TensorView {
//        queue(#function, {}, &result) {
//            for index in $1.indices { $1[index] = value }
//        }
    }
    
    //--------------------------------------------------------------------------
    /// fillWithIndex(x:startAt:
    func fillWithIndex<T>(_ result: inout T, startAt: Int) where
        T: TensorView, T.Values.Element: AnyNumeric
    {
//        queue(#function, {}, &result) {
//            var value = startAt
//            for index in $1.indices {
//                $1[index] = T.Values.Element(any: value)
//                value += 1
//            }
//        }
    }

    //--------------------------------------------------------------------------
    /// floor
    func floor<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.map(to: &$1) {
//                T.Values.Element(any: Foundation.floorf($0.asFloat))
//            }
//        }
    }
    
    //--------------------------------------------------------------------------
    /// greater
    func greater<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Values.Element: Comparable
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) { $0 > $1 }
//        }
    }
    
    //--------------------------------------------------------------------------
    /// greaterOrEqual
    func greaterOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Values.Element: Comparable
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) { $0 >= $1 }
//        }
    }

    //--------------------------------------------------------------------------
    /// less
    func less<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Values.Element: Comparable
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) { $0 < $1 }
//        }
    }

    //--------------------------------------------------------------------------
    /// lessOrEqual
    func lessOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Values.Element: Comparable
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) { $0 <= $1 }
//        }
    }

    //--------------------------------------------------------------------------
    /// log(x:result:
    func log<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
        queue(#function, { try x.values() }, &result) {
            $0.map(to: &$1) {
                T.Values.Element(any: Foundation.log($0.asFloat))
            }
        }
    }
    
    //--------------------------------------------------------------------------
    /// logicalNot(x:result:
    func logicalNot<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element == Bool
    {
//        queue(#function, { try x.values() }, &result) { $0.map(to: &$1) { !$0 } }
    }
    
    //--------------------------------------------------------------------------
    /// logicalAnd(x:result:
    func logicalAnd<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Values.Element == Bool
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) { $0 && $1 }
//        }
    }

    //--------------------------------------------------------------------------
    /// logicalOr(x:result:
    func logicalOr<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Values.Element == Bool
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) { $0 || $1 }
//        }
    }

    //--------------------------------------------------------------------------
    /// logSoftmax(x:result:
    func logSoftmax<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    /// matmul(lhs:rhs:result:
    func matmul<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Values.Element: Numeric
    {
        
    }
    
    //--------------------------------------------------------------------------
    /// max
    func max<T>(x: T, along axes: [Int], result: inout T) where
        T: TensorView, T.Values.Element: Numeric
    {
        
    }
    
    //--------------------------------------------------------------------------
    /// maximum(lhs:rhs:result:
    func maximum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Values.Element: Comparable
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) { $0 >= $1 ? $0 : $1 }
//        }
    }
    
    //--------------------------------------------------------------------------
    /// mean
    func mean<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Values.Element: Numeric
    {
        
    }
    
    //--------------------------------------------------------------------------
    /// min
    func min<T>(x: T, along axes: [Int], result: inout T) where
        T: TensorView, T.Values.Element: Numeric
    {
        
    }
    
    //--------------------------------------------------------------------------
    /// minimum(lhs:rhs:result:
    func minimum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Values.Element: Comparable
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) { $0 <= $1 ? $0 : $1 }
//        }
    }
    
    //--------------------------------------------------------------------------
    // mod
    func mod<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) {
//                T.Values.Element(any: fmodf($0.asFloat, $1.asFloat))
//            }
//        }
    }

    //--------------------------------------------------------------------------
    // mul
    func mul<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Values.Element: Numeric
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) { $0 * $1 }
//        }
    }
    
    //--------------------------------------------------------------------------
    // neg
    func neg<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: SignedNumeric
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.map(to: &$1) { -$0 }
//        }
    }
    
    //--------------------------------------------------------------------------
    // notEqual
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Values.Element: Equatable
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) { $0 != $1 }
//        }
    }

    //--------------------------------------------------------------------------
    // pow
    // TODO there needs to be a generic math library!
    func pow<T>(x: T, y: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyNumeric
    {
//        queue(#function, { try (x.values(), y.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) {
//                T.Values.Element(any: Foundation.pow($0.asDouble, $1.asDouble))
//            }
//        }
    }

    //--------------------------------------------------------------------------
    // prod
    func prod<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Values.Element: AnyNumeric
    {
//        let one = T.Values.Element(any: 1)
//        queue(#function, { try x.values() }, &result) {
//            $0.reduce(to: &$1, one) { $0 * $1 }
//        }
    }
    
    //--------------------------------------------------------------------------
    // rsqrt
    func rsqrt<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.map(to: &$1) { 1 / Foundation.sqrt($0) }
//        }
    }

    //--------------------------------------------------------------------------
    // replacing
    func replacing<T>(x: T, with other: T, where mask: T.BoolView,
                      result: inout T) where T: TensorView
    {
//        queue(#function, { try (x.values(), other.values(), mask.values()) },
//              &result)
//        {
//            zip($0.0, $0.1, $0.2).map(to: &$1) { $2 ? $1 : $0 }
//        }
    }
    
    //--------------------------------------------------------------------------
    // sin
    func sin<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.map(to: &$1) {
//                T.Values.Element(any: Foundation.sinf($0.asFloat))
//            }
//        }
    }

    
    //--------------------------------------------------------------------------
    // sinh
    func sinh<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.map(to: &$1) {
//                T.Values.Element(any: Foundation.sinhf($0.asFloat))
//            }
//        }
    }

    //--------------------------------------------------------------------------
    // square
    func square<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: Numeric
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.map(to: &$1) { $0 * $0 }
//        }
    }
    
    //--------------------------------------------------------------------------
    // squaredDifference
    func squaredDifference<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Values.Element: Numeric
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) {
//                let diff = $0 - $1
//                return diff * diff
//            }
//        }
    }
    
    //--------------------------------------------------------------------------
    // sqrt
    func sqrt<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.map(to: &$1) { Foundation.sqrt($0) }
//        }
    }
    
    //--------------------------------------------------------------------------
    // subtract
    func subtract<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Values.Element: Numeric
    {
//        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
//            zip($0.0, $0.1).map(to: &$1) { $0 - $1 }
//        }
    }
    
    //--------------------------------------------------------------------------
    // sum
    func sum<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Values.Element: Numeric
    {
//        if let axes = axes, axes.shape.extents[0] > 0 {
//            assert(axes.rank <= x.rank, "rank mismatch")
//            queue(#function, { try (x.values(), axes.values()) }, &result) { params, result in
//
//            }
//        } else {
//            queue(#function, { try x.values() }, &result) {
//                $0.reduce(to: &$1, T.Values.Element.zero) { $0 + $1 }
//            }
//        }
    }
    
    //--------------------------------------------------------------------------
    // tan
    func tan<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.map(to: &$1) {
//                T.Values.Element(any: Foundation.tanf($0.asFloat))
//            }
//        }
    }

    //--------------------------------------------------------------------------
    // tanh
    func tanh<T>(x: T, result: inout T) where
        T: TensorView, T.Values.Element: AnyFloatingPoint
    {
//        queue(#function, { try x.values() }, &result) {
//            $0.map(to: &$1) {
//                T.Values.Element(any: Foundation.tanhf($0.asFloat))
//            }
//        }
    }
}
