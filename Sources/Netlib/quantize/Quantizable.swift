//******************************************************************************
//  Created by Edward Connell on 5/18/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// Quantizable
public protocol Quantizable {
    static var normalScale: Float { get }
    static var normalInverseScale: Float { get }
    init(value: Quantizable)
    init(value: Quantizable, scale: Quantizable, bias: Quantizable)
    
    //--------------------------------------------------------------------------
    // scalar types
    var asUInt8: UInt8 { get }
    var asFloat: Float { get }
    
    func toUInt8(_ scale: Quantizable, _ bias: Quantizable) -> UInt8
    func toFloat(_ scale: Quantizable, _ bias: Quantizable) -> Float
    
    //--------------------------------------------------------------------------
    // short vector types
    var asRGBUInt8: RGB<UInt8> { get }
    var asRGBFloat: RGB<Float> { get }
    
    func toRGBUInt8(_ scale: Quantizable, _ bias: Quantizable) -> RGB<UInt8>
    func toRGBFloat(_ scale: Quantizable, _ bias: Quantizable) -> RGB<Float>
}

//==============================================================================
/// Quantizable extensions
public extension Quantizable {
    /// converts a scalar value Int --> Float
    @inlinable @inline(__always)
    func convert<I, F>(integer value: I, _ scale: F, _ bias: F) -> F
        where I: FixedWidthInteger, F: BinaryFloatingPoint
    {
        if value == 0 {
            return F(bias)
        } else if value > 0 {
            return (F(value) + 1) * scale + bias
        } else {
            return F(value) * scale + bias
        }
    }
    
    /// converts a scalar value Float --> Int
    @inlinable @inline(__always)
    func convert<F, I>(floating value: F, _ scale: F, _ bias: F) -> I
        where F: BinaryFloatingPoint, I: FixedWidthInteger
    {
        if value == bias {
            return 0
        } else if value > 0 {
            return I((value - bias) * scale - 1)
        } else {
            return I((value - bias) * scale)
        }
    }
    
    //--------------------------------------------------------------------------
    // short vector conversion from scalars
    var asRGBUInt8: RGB<UInt8> {
        return RGB<UInt8>(r: asUInt8, g: asUInt8, b: asUInt8)
    }
    
    var asRGBFloat: RGB<Float> {
        return RGB<Float>(r: asFloat, g: asFloat, b: asFloat)
    }
    
    func toRGBUInt8(_ scale: Quantizable,_ bias: Quantizable) -> RGB<UInt8> {
        return RGB<UInt8>(r: toUInt8(scale, bias),
                          g: toUInt8(scale, bias),
                          b: toUInt8(scale, bias))
    }
    
    func toRGBFloat(_ scale: Quantizable, _ bias: Quantizable) -> RGB<Float> {
        return RGB<Float>(r: toFloat(scale, bias),
                          g: toFloat(scale, bias),
                          b: toFloat(scale, bias))
    }
}

//==============================================================================
// UInt8
extension UInt8: Quantizable {
    // properties
    public static var normalScale: Float { return 1 / normalInverseScale }
    public static var normalInverseScale: Float { return Float(UInt8.max) + 1 }

    // initializers
    @inlinable @inline(__always)
    public init(value: Quantizable) { self = value.asUInt8 }

    @inlinable @inline(__always)
    public init(value: Quantizable, scale: Quantizable, bias: Quantizable) {
        self = value.toUInt8(scale, bias)
    }
    
    //--------------------------------------------------------------------------
    // scalar conversion
    @inlinable @inline(__always)
    public func toUInt8(_ scale: Quantizable, _ bias: Quantizable) -> UInt8 {
        return self
    }
    
    @inlinable @inline(__always)
    public func toFloat(_ scale: Quantizable, _ bias: Quantizable) -> Float {
        let transformScale = UInt8.normalScale * scale.asFloat
        return convert(integer: self, transformScale, bias.asFloat)
    }
}

//==============================================================================
// Float
extension Float: Quantizable {
    // properties
    public static var normalScale: Float { return 1  }
    public static var normalInverseScale: Float { return 1 }

    // initializers
    @inlinable @inline(__always)
    public init(value: Quantizable) { self = value.asFloat }
    
    @inlinable @inline(__always)
    public init(value: Quantizable, scale: Quantizable, bias: Quantizable) {
        self = value.toFloat(scale, bias)
    }
    
    //--------------------------------------------------------------------------
    // scalar conversion
    @inlinable @inline(__always)
    public func toUInt8(_ scale: Quantizable, _ bias: Quantizable) -> UInt8 {
        let transformScale = UInt8.normalScale * scale.asFloat
        return convert(floating: self, transformScale, bias.asFloat)
    }
    
    @inlinable @inline(__always)
    public func toFloat(_ scale: Quantizable, _ bias: Quantizable) -> Float {
        return self
    }
}

//==============================================================================
// RGB
extension RGB: Quantizable where Component == UInt8 {
    // the scales are unused
    public static var normalScale: Float { return 1 }
    public static var normalInverseScale: Float { return 1 }

    // initializers
    @inlinable @inline(__always)
    public init(value: Quantizable) { self = value.asRGBUInt8 }
    
    @inlinable @inline(__always)
    public init(value: Quantizable, scale: Quantizable, bias: Quantizable) {
        self = value.toRGBUInt8(scale, bias)
    }
    
    public var asUInt8: UInt8 {
        // TODO: grayscale convert
        fatalError("not implemented yet")
    }
    
    public var asFloat: Float {
        // TODO: grayscale convert
        fatalError("not implemented yet")
    }
    
    //--------------------------------------------------------------------------
    // scalar conversion
    @inlinable @inline(__always)
    public func toUInt8(_ scale: Quantizable, _ bias: Quantizable) -> UInt8 {
        // TODO: grayscale convert
        fatalError("not implemented yet")
    }
    
    @inlinable @inline(__always)
    public func toFloat(_ scale: Quantizable, _ bias: Quantizable) -> Float {
        // TODO: grayscale convert
        fatalError("not implemented yet")
    }
    
    //--------------------------------------------------------------------------
    // short vector conversion
    @inlinable @inline(__always)
    public var asRGBUInt8: RGB<UInt8> {
        return self
    }
    
    @inlinable @inline(__always)
    public var asRGBFloat: RGB<Float> {
        return RGB<Float>(r: r.asFloat, g: g.asFloat, b: b.asFloat)
    }
    
    @inlinable @inline(__always)
    public func toRGBUInt8(_ scale: Quantizable,
                           _ bias: Quantizable) -> RGB<UInt8> {
        return RGB<UInt8>(r: r.toUInt8(scale, bias),
                          g: g.toUInt8(scale, bias),
                          b: b.toUInt8(scale, bias))
    }
    
    @inlinable @inline(__always)
    public func toRGBFloat(_ scale: Quantizable,
                           _ bias: Quantizable) -> RGB<Float> {
        return RGB<Float>(r: r.toFloat(scale, bias),
                          g: g.toFloat(scale, bias),
                          b: b.toFloat(scale, bias))
    }

}

