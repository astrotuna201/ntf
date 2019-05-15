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
    associatedtype Element
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

    /// converts the tensor element value type to the viewed value type
    func convert(element: Element) -> Viewed
    /// converts the tensor viewed value type to the element value type
    func convert(viewed: Viewed) -> Element
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
    Element: UniformDenseScalar,
    Element.Component: FixedWidthInteger
{
    mutating func updateScales() {
        _inverseTransformScale = (Float(Element.Component.max) + 1) * scale
        _transformScale = 1 / _inverseTransformScale
    }
}

//==============================================================================
/// Quantize1
/// used to convert numeric scalars
public struct Quantize1<Element, Viewed>: Quantizer where
    Element: FixedWidthInteger,
    Viewed: BinaryFloatingPoint
{
    public var bias: Float
    public var scale: Float { didSet { updateScales() } }
    public var _transformScale: Float = 1
    public var _inverseTransformScale: Float = 1
    
    public init(scale: Float = 1, bias: Float = 0) {
        self.scale = scale
        self.bias = bias
        updateScales()
    }
    
    private mutating func updateScales() {
        _inverseTransformScale = (Float(Element.max) + 1) * scale
        _transformScale = 1 / _inverseTransformScale
    }
    
    @inlinable @inline(__always)
    public func convert(element: Element) -> Viewed {
        return convert(value: element)
    }
    
    @inlinable @inline(__always)
    public func convert(viewed: Viewed) -> Element {
        return convert(value: viewed)
    }
}

//==============================================================================
/// Quantize2
/// used to convert short vector types with 2 numeric scalars
public struct Quantize2<Element, Viewed>: Quantizer where
    Element: UniformDenseScalar2, Element.Component: FixedWidthInteger,
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
    public func convert(element: Element) -> Viewed {
        return Viewed(c0: convert(value: element.c0),
                      c1: convert(value: element.c1))
    }
    
    @inlinable @inline(__always)
    public func convert(viewed: Viewed) -> Element {
        return Element(c0: convert(value: viewed.c0),
                      c1: convert(value: viewed.c1))
    }
}

//==============================================================================
/// Quantize3
/// used to convert short vector types with 3 numeric scalars
public struct Quantize3<Element, Viewed>: Quantizer where
    Element: UniformDenseScalar3, Element.Component: FixedWidthInteger,
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
    public func convert(element: Element) -> Viewed {
        return Viewed(c0: convert(value: element.c0),
                      c1: convert(value: element.c1),
                      c2: convert(value: element.c2))
    }
    
    @inlinable @inline(__always)
    public func convert(viewed: Viewed) -> Element {
        return Element(c0: convert(value: viewed.c0),
                      c1: convert(value: viewed.c1),
                      c2: convert(value: viewed.c2))
    }
}

//==============================================================================
/// Quantize4
/// used to convert short vector types with 4 numeric scalars
public struct Quantize4<Element, Viewed>: Quantizer where
    Element: UniformDenseScalar4, Element.Component: FixedWidthInteger,
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
    public func convert(element: Element) -> Viewed {
        return Viewed(c0: convert(value: element.c0),
                      c1: convert(value: element.c1),
                      c2: convert(value: element.c2),
                      c3: convert(value: element.c3))
    }
    
    @inlinable @inline(__always)
    public func convert(viewed: Viewed) -> Element {
        return Element(c0: convert(value: viewed.c0),
                      c1: convert(value: viewed.c1),
                      c2: convert(value: viewed.c2),
                      c3: convert(value: viewed.c3))
    }
}

