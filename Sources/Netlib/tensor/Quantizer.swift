//==============================================================================
/// QuantizerProtocol
public protocol QuantizerProtocol {
    associatedtype Stored
    associatedtype Viewed
    
    /// the bias to apply during conversion
    var bias: Float { get set }
    /// the scale to apply during conversion
    var scale: Float { get set }
    /// the scale factor used to map into the range of the type, times
    /// the user scale factor
    var _transformScale: Float { get set }
    /// a private scale factor used by the transform functions
    var _oneOverTransformScale: Float { get set }

    //--------------------------------------------------------------------------
    /// converts from Scalar to ViewedScalar
    func convert(stored: Stored) -> Viewed
    /// converts from Scalar to ViewedScalar
    func convert(viewed: Viewed) -> Stored
}

//==============================================================================
/// QuantizerProtocol extensions
public extension QuantizerProtocol {
    var typeNormalScale: Float { fatalError("not implemented") }
}

public extension QuantizerProtocol where Stored: FixedWidthInteger {
    var typeNormalScale: Float {
        return Float(1.0) / (Float(Stored.max) + 1)
    }
}

public extension QuantizerProtocol where Viewed: FixedWidthInteger {
    var typeNormalScale: Float {
        return Float(1.0) / (Float(Viewed.max) + 1)
    }
}

//==============================================================================
/// Quantizer
public struct Quantizer<Stored, Viewed>: QuantizerProtocol {
    // properties
    public var _transformScale: Float = 0
    public var _oneOverTransformScale: Float = 0
    public var bias: Float = 0
    public var scale: Float = 1 { didSet { updateScales() } }

    // initializers
    public init() {
        updateScales()
    }
    public init(scale: Float, bias: Float) {
        self.scale = scale
        self.bias = bias
        updateScales()
    }
    
    private mutating func updateScales() {
        _transformScale = typeNormalScale * scale
        _oneOverTransformScale = 1 / _transformScale
    }
}

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
    func Float2Int<TF, TI>(value: TF) -> TI
        where TF: BinaryFloatingPoint, TI: BinaryInteger
    {
        if value == TF(bias) {
            return 0
        } else if value > 0 {
            return TI(((Float(value) - bias) * _oneOverTransformScale) - 1)
        } else {
            return TI((Float(value) - bias) * _oneOverTransformScale)
        }
    }

    //==========================================================================
    @inlinable @inline(__always)
    func Int2Float<TI, TF>(value: TI) -> TF
        where TI: BinaryInteger, TF: BinaryFloatingPoint
    {
        if value == 0 {
            return TF(bias)
        } else if value > 0 {
            return TF((Float(value) + 1) * _transformScale + bias)
        } else {
            return TF((Float(value)) * _transformScale + bias)
        }
    }
}

//==============================================================================
/// Integer --> Float
public extension QuantizerProtocol where
    Stored: BinaryInteger, Viewed: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Int2Float(value: stored)
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Float2Int(value: viewed)
    }
}

//==============================================================================
/// Float --> Integer -->
public extension QuantizerProtocol where
    Stored: BinaryFloatingPoint, Viewed: BinaryInteger
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Float2Int(value: stored)
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Int2Float(value: viewed)
    }
}

//==============================================================================
/// UniformDenseScalar2
public extension QuantizerProtocol where
    Stored: UniformDenseScalar2, Stored.Component: BinaryInteger,
    Viewed: UniformDenseScalar2, Viewed.Component: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: Int2Float(value: stored.c0),
                      c1: Int2Float(value: stored.c1))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: Float2Int(value: viewed.c0),
                      c1: Float2Int(value: viewed.c1))
    }
}

public extension QuantizerProtocol where
    Stored: UniformDenseScalar2, Stored.Component: BinaryFloatingPoint,
    Viewed: UniformDenseScalar2, Viewed.Component: BinaryInteger
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: Float2Int(value: stored.c0),
                      c1: Float2Int(value: stored.c1))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: Int2Float(value: viewed.c0),
                      c1: Int2Float(value: viewed.c1))
    }
}
//==============================================================================
/// UniformDenseScalar3
public extension QuantizerProtocol where
    Stored: UniformDenseScalar3, Stored.Component: BinaryInteger,
    Viewed: UniformDenseScalar3, Viewed.Component: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: Int2Float(value: stored.c0),
                      c1: Int2Float(value: stored.c1),
                      c2: Int2Float(value: stored.c2))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: Float2Int(value: viewed.c0),
                      c1: Float2Int(value: viewed.c1),
                      c2: Float2Int(value: viewed.c2))
    }
}

public extension QuantizerProtocol where
    Stored: UniformDenseScalar3, Stored.Component: BinaryFloatingPoint,
    Viewed: UniformDenseScalar3, Viewed.Component: BinaryInteger
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: Float2Int(value: stored.c0),
                      c1: Float2Int(value: stored.c1),
                      c2: Float2Int(value: stored.c2))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: Int2Float(value: viewed.c0),
                      c1: Int2Float(value: viewed.c1),
                      c2: Int2Float(value: viewed.c2))
    }
}

//==============================================================================
/// UniformDenseScalar4
public extension QuantizerProtocol where
    Stored: UniformDenseScalar4, Stored.Component: BinaryInteger,
    Viewed: UniformDenseScalar4, Viewed.Component: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: Int2Float(value: stored.c0),
                      c1: Int2Float(value: stored.c1),
                      c2: Int2Float(value: stored.c2),
                      c3: Int2Float(value: stored.c3))
    }

    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: Float2Int(value: viewed.c0),
                      c1: Float2Int(value: viewed.c1),
                      c2: Float2Int(value: viewed.c2),
                      c3: Float2Int(value: viewed.c3))
    }
}

public extension QuantizerProtocol where
    Stored: UniformDenseScalar4, Stored.Component: BinaryFloatingPoint,
    Viewed: UniformDenseScalar4, Viewed.Component: BinaryInteger
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: Float2Int(value: stored.c0),
                      c1: Float2Int(value: stored.c1),
                      c2: Float2Int(value: stored.c2),
                      c3: Float2Int(value: stored.c3))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: Int2Float(value: viewed.c0),
                      c1: Int2Float(value: viewed.c1),
                      c2: Int2Float(value: viewed.c2),
                      c3: Int2Float(value: viewed.c3))
    }
}

