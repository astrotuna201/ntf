//==============================================================================
/// QuantizerProtocol
public protocol QuantizerProtocol {
    associatedtype Stored: Quantizable
    associatedtype Viewed: Quantizable
    
    /// the bias to apply during conversion
    var bias: Float { get set }
    /// the scale to apply during conversion
    var scale: Float { get set }
    /// the scale factor used to map into the range of the type, times
    /// the user scale factor
    var _transformScale: Float { get }
    /// a private scale factor used by the transform functions
    var _inverseTransformScale: Float { get }

    //--------------------------------------------------------------------------
    /// converts from Scalar to ViewedScalar
    func convert(stored: Stored) -> Viewed
    /// converts from Scalar to ViewedScalar
    func convert(viewed: Viewed) -> Stored
}

//==============================================================================
/// Quantizer
/// an object used to perform value conversion between types
public struct Quantizer<Stored, Viewed>: QuantizerProtocol
    where Stored: Quantizable, Viewed: Quantizable
{
    // properties
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
        _transformScale = Stored.normalScale * Viewed.normalScale * scale
        _inverseTransformScale = 1 / _transformScale
    }
}

//==============================================================================
/// Quantizable
/// conformed to by convertable types
public protocol Quantizable {
    static var normalScale: Float { get }
}

public extension Quantizable {
    static var normalScale: Float { return 1 }
}

public extension Quantizable where Self: FixedWidthInteger {
    static var normalScale: Float { return 1 / (Float(self.max) + 1) }
}

public extension Quantizable where Self: UniformDenseScalar {
    static var normalScale: Float { return 1 }
}

public extension Quantizable where
    Self: UniformDenseScalar, Component: Quantizable {
    static var normalScale: Float { return Component.normalScale }
}

extension Int8: Quantizable {}
extension UInt8: Quantizable {}
extension Int16: Quantizable {}
extension UInt16: Quantizable {}
extension Int32: Quantizable {}
extension UInt32: Quantizable {}
extension Float: Quantizable {}
extension Double: Quantizable {}

//==============================================================================
/// QuantizerProtocol
public extension QuantizerProtocol {
    func convert(stored: Stored) -> Viewed { fatalError("not implemented") }
    func convert(viewed: Viewed) -> Stored { fatalError("not implemented") }

    //==========================================================================
    /// NOTE: It's likely most of the time value will be Float, so the cast of
    /// TF(bias) should be thrown out by the compiler. In the case of
    /// something like Float16, it will likely be faster to cast anyway.
    @inlinable @inline(__always)
    func convert<T, R>(value: T) -> R
        where T: BinaryFloatingPoint, R: FixedWidthInteger
    {
        if value == T(bias) {
            return 0
        } else if value > 0 {
            return R(((Float(value) - bias) * _inverseTransformScale) - 1)
        } else {
            return R((Float(value) - bias) * _inverseTransformScale)
        }
    }

    //==========================================================================
    @inlinable @inline(__always)
    func convert<T, R>(value: T) -> R
        where T: FixedWidthInteger, R: BinaryFloatingPoint
    {
        if value == 0 {
            return R(bias)
        } else if value > 0 {
            return R((Float(value) + 1) * _transformScale + bias)
        } else {
            return R((Float(value)) * _transformScale + bias)
        }
    }
}

//==============================================================================
/// Integer --> Float
public extension QuantizerProtocol where
    Stored: FixedWidthInteger, Viewed: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return convert(value: stored)
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return convert(value: viewed)
    }
}

//==============================================================================
/// Float --> Integer -->
public extension QuantizerProtocol where
    Stored: BinaryFloatingPoint, Viewed: FixedWidthInteger
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return convert(value: stored)
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return convert(value: viewed)
    }
}

//==============================================================================
/// UniformDenseScalar2
public extension QuantizerProtocol where
    Stored: UniformDenseScalar2, Stored.Component: FixedWidthInteger,
    Viewed: UniformDenseScalar2, Viewed.Component: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: convert(value: stored.c0),
                      c1: convert(value: stored.c1))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: convert(value: viewed.c0),
                      c1: convert(value: viewed.c1))
    }
}

public extension QuantizerProtocol where
    Stored: UniformDenseScalar2, Stored.Component: BinaryFloatingPoint,
    Viewed: UniformDenseScalar2, Viewed.Component: FixedWidthInteger
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: convert(value: stored.c0),
                      c1: convert(value: stored.c1))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: convert(value: viewed.c0),
                      c1: convert(value: viewed.c1))
    }
}

//==============================================================================
/// UniformDenseScalar3
public extension QuantizerProtocol where
    Stored: UniformDenseScalar3, Stored.Component: FixedWidthInteger,
    Viewed: UniformDenseScalar3, Viewed.Component: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: convert(value: stored.c0),
                      c1: convert(value: stored.c1),
                      c2: convert(value: stored.c2))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: convert(value: viewed.c0),
                      c1: convert(value: viewed.c1),
                      c2: convert(value: viewed.c2))
    }
}

public extension QuantizerProtocol where
    Stored: UniformDenseScalar3, Stored.Component: BinaryFloatingPoint,
    Viewed: UniformDenseScalar3, Viewed.Component: FixedWidthInteger
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: convert(value: stored.c0),
                      c1: convert(value: stored.c1),
                      c2: convert(value: stored.c2))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: convert(value: viewed.c0),
                      c1: convert(value: viewed.c1),
                      c2: convert(value: viewed.c2))
    }
}

//==============================================================================
/// UniformDenseScalar4
public extension QuantizerProtocol where
    Stored: UniformDenseScalar4, Stored.Component: FixedWidthInteger,
    Viewed: UniformDenseScalar4, Viewed.Component: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: convert(value: stored.c0),
                      c1: convert(value: stored.c1),
                      c2: convert(value: stored.c2),
                      c3: convert(value: stored.c3))
    }

    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: convert(value: viewed.c0),
                      c1: convert(value: viewed.c1),
                      c2: convert(value: viewed.c2),
                      c3: convert(value: viewed.c3))
    }
}

public extension QuantizerProtocol where
    Stored: UniformDenseScalar4, Stored.Component: BinaryFloatingPoint,
    Viewed: UniformDenseScalar4, Viewed.Component: FixedWidthInteger
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: convert(value: stored.c0),
                      c1: convert(value: stored.c1),
                      c2: convert(value: stored.c2),
                      c3: convert(value: stored.c3))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: convert(value: viewed.c0),
                      c1: convert(value: viewed.c1),
                      c2: convert(value: viewed.c2),
                      c3: convert(value: viewed.c3))
    }
}

