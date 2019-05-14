//******************************************************************************
//  Created by Edward Connell on 5/1/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// Quantizer
/// defines a quantizer converting object
public protocol Quantizer {
    // types
    associatedtype Stored
    associatedtype Viewed
    
    // properties
    /// bias applied to Viewed value
    var bias: Float { get set }
    /// scale applied to Viewed value
    var scale: Float { get set }
    /// integer to float transform scale
    var _transformScale: Float { get set }
    /// float to integer transform scale
    var _inverseTransformScale: Float { get set }

    /// converts the tensor stored value type to the viewed value type
    func convert(stored: Stored) -> Viewed
    /// converts the tensor viewed value type to the stored value type
    func convert(viewed: Viewed) -> Stored
}

//==============================================================================
/// Quantizer extensions
public extension Quantizer {
    /// converts a scalar value Int --> Float
    @inlinable @inline(__always)
    func convert<Value, Result>(value: Value) -> Result
        where Value: FixedWidthInteger, Result: BinaryFloatingPoint
    {
        if value == 0 {
            return Result(bias)
        } else if value > 0 {
            return Result((Float(value) + 1) * _transformScale + bias)
        } else {
            return Result(Float(value) * _transformScale + bias)
        }
    }
    
    /// converts a scalar value Float --> Int
    @inlinable @inline(__always)
    func convert<Value, Result>(value: Value) -> Result
        where Value: BinaryFloatingPoint, Result: FixedWidthInteger
    {
        let viewed = Float(value)
        if viewed == bias {
            return 0
        } else if viewed > 0 {
            return Result((viewed - bias) * _inverseTransformScale - 1)
        } else {
            return Result((viewed - bias) * _inverseTransformScale)
        }
    }
}

public extension Quantizer where
    Stored: UniformDenseScalar,
    Stored.Component: FixedWidthInteger
{
    mutating func updateScales() {
        _inverseTransformScale = (Float(Stored.Component.max) + 1) * scale
        _transformScale = 1 / _inverseTransformScale
    }
}

//==============================================================================
/// Quantize1
/// used to convert numeric scalars
public struct Quantize1<Stored, Viewed>: Quantizer where
    Stored: FixedWidthInteger,
    Viewed: BinaryFloatingPoint
{
    public var bias: Float
    public var scale: Float { didSet { updateScales() } }
    public var _transformScale: Float = 1
    public var _inverseTransformScale: Float = 1
    
    // initializers
    public init(scale: Float = 1, bias: Float = 0) {
        self.scale = scale
        self.bias = bias
        updateScales()
    }
    
    private mutating func updateScales() {
        _inverseTransformScale = (Float(Stored.max) + 1) * scale
        _transformScale = 1 / _inverseTransformScale
    }
    
    @inlinable @inline(__always)
    public func convert(stored: Stored) -> Viewed {
        return convert(value: stored)
    }
    
    @inlinable @inline(__always)
    public func convert(viewed: Viewed) -> Stored {
        return convert(value: viewed)
    }
}

//==============================================================================
/// Quantize2
/// used to convert short vector types with 2 numeric scalars
public struct Quantize2<Stored, Viewed>: Quantizer where
    Stored: UniformDenseScalar2, Stored.Component: FixedWidthInteger,
    Viewed: UniformDenseScalar2, Viewed.Component: BinaryFloatingPoint
{
    public var bias: Float
    public var scale: Float { didSet { updateScales() } }
    public var _transformScale: Float = 1
    public var _inverseTransformScale: Float = 1
    
    // initializers
    public init(scale: Float = 1, bias: Float = 0) {
        self.scale = scale
        self.bias = bias
        updateScales()
    }
    
    @inlinable @inline(__always)
    public func convert(stored: Stored) -> Viewed {
        return Viewed(c0: convert(value: stored.c0),
                      c1: convert(value: stored.c1))
    }
    
    @inlinable @inline(__always)
    public func convert(viewed: Viewed) -> Stored {
        return Stored(c0: convert(value: viewed.c0),
                      c1: convert(value: viewed.c1))
    }
}

//==============================================================================
/// Quantize3
/// used to convert short vector types with 3 numeric scalars
public struct Quantize3<Stored, Viewed>: Quantizer where
    Stored: UniformDenseScalar3, Stored.Component: FixedWidthInteger,
    Viewed: UniformDenseScalar3, Viewed.Component: BinaryFloatingPoint
{
    public var bias: Float
    public var scale: Float { didSet { updateScales() } }
    public var _transformScale: Float = 1
    public var _inverseTransformScale: Float = 1
    
    // initializers
    public init(scale: Float = 1, bias: Float = 0) {
        self.scale = scale
        self.bias = bias
        updateScales()
    }
    
    @inlinable @inline(__always)
    public func convert(stored: Stored) -> Viewed {
        return Viewed(c0: convert(value: stored.c0),
                      c1: convert(value: stored.c1),
                      c2: convert(value: stored.c2))
    }
    
    @inlinable @inline(__always)
    public func convert(viewed: Viewed) -> Stored {
        return Stored(c0: convert(value: viewed.c0),
                      c1: convert(value: viewed.c1),
                      c2: convert(value: viewed.c2))
    }
}

//==============================================================================
/// Quantize4
/// used to convert short vector types with 4 numeric scalars
public struct Quantize4<Stored, Viewed>: Quantizer where
    Stored: UniformDenseScalar4, Stored.Component: FixedWidthInteger,
    Viewed: UniformDenseScalar4, Viewed.Component: BinaryFloatingPoint
{
    public var bias: Float
    public var scale: Float { didSet { updateScales() } }
    public var _transformScale: Float = 1
    public var _inverseTransformScale: Float = 1
    
    // initializers
    public init(scale: Float = 1, bias: Float = 0) {
        self.scale = scale
        self.bias = bias
        updateScales()
    }
    
    @inlinable @inline(__always)
    public func convert(stored: Stored) -> Viewed {
        return Viewed(c0: convert(value: stored.c0),
                      c1: convert(value: stored.c1),
                      c2: convert(value: stored.c2),
                      c3: convert(value: stored.c3))
    }
    
    @inlinable @inline(__always)
    public func convert(viewed: Viewed) -> Stored {
        return Stored(c0: convert(value: viewed.c0),
                      c1: convert(value: viewed.c1),
                      c2: convert(value: viewed.c2),
                      c3: convert(value: viewed.c3))
    }
}

