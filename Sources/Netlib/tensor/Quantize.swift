//******************************************************************************
//  Created by Edward Connell on 5/21/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//@inlinable @inline(__always)
//func convert<Value, Result>(value: Value) -> Result
//    where Value: FixedWidthInteger, Result: BinaryFloatingPoint
//{
//    if value == 0 {
//        return Result(bias)
//    } else if value > 0 {
//        return Result((Float(value) + 1) * _transformScale + bias)
//    } else {
//        return Result(Float(value) * _transformScale + bias)
//    }
//}
//
//@inlinable @inline(__always)
//func convert<Value, Result>(value: Value) -> Result
//    where Value: BinaryFloatingPoint, Result: FixedWidthInteger
//{
//    let viewed = Float(value)
//    if viewed == bias {
//        return 0
//    } else if viewed > 0 {
//        return Result((viewed - bias) * _inverseTransformScale - 1)
//    } else {
//        return Result((viewed - bias) * _inverseTransformScale)
//    }
//}

public extension TensorView where Element: FixedWidthInteger {
    /// computes the mean and stddev of x, scales by `stddevs`,
    /// and converts to Element
    init<T>(quantizing x: T, stddevs: Float, name: String? = nil,
            using stream: DeviceStream? = nil) where
        T: TensorView, T.Element: BinaryFloatingPoint
    {
        let (xmean, xstd) = x.standardDeviation()
        let shape = x.shape.dense
        let array = TensorArray(type: Element.self,
                                count: shape.elementCount,
                                name: name ?? String(describing: Self.self))
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: nil, traversal: .normal, isShared: false)
        do {
            let bias = try T.Element(xmean.scalarValue())
            let std =  try T.Element(xstd.scalarValue())
            let inverseTransformScale = T.Element(Element.max + 1) * std
//            let transformScale = 1 / inverseTransformScale
            let values = try x.values()
            var results = try self.mutableValues()
            for (i, j) in zip(results.indices, values.indices) {
                let value = values[j]
                if value == bias {
                    results[i] = 0
                } else if value > 0 {
                    results[i] = Element((value - bias) * inverseTransformScale - 1)
                } else {
                    results[i] = Element((value - bias) * inverseTransformScale)
                }
                results[i] = Element()
            }
        } catch {
            print(String(describing: error))
        }
    }

    /// computes the mmin/max of x, then scales and converts to Element
    init<T>(quantizing x: T)
        where T: TensorView, T.Element: FloatingPoint
    {
        fatalError()
    }
}

public extension TensorView where Element: FloatingPoint {
    /// converts x to Element by applying scale * normalScale + bias
    init<T>(quantizing x: T, scale: Float, bias: Float)
        where T: TensorView, T.Element: FixedWidthInteger
    {
        fatalError()
    }
}
