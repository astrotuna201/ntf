//******************************************************************************
//  Created by Edward Connell on 5/21/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

public extension TensorView where Element: FixedWidthInteger {
    /// computes the mean and stddev of x, scales by `stddevs`,
    /// and converts to Element
//    init<T>(quantizing x: T, stddevs: Float,
//            name: String? = nil,
//            using stream: DeviceStream? = nil) throws
//        where T: TensorView, T.Element: BinaryFloatingPoint
//    {
//        var bias = x.createDense(with: [1])
//        var std = x.createDense(with: [1])
//        Netlib.standardDeviation(x, meanValue: &bias, result: &std)
//        let scale = try std.readOnly()[0] * T.Element(stddevs)
//        let shape = x.shape.dense
////        let array = TensorArray(type: Element.self,
////                                count: shape.elementCount,
////                                name: name ?? String(describing: Self.self))
////        self.init(shape: shape, dataShape: shape,
////                  tensorArray: array, viewDataOffset: 0,
////                  indexAlignment: nil, traversal: .normal, isShared: false)
//        self = (x - bias) * scale - 1
//    }

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
