//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// conformance indicates that scalar components are of the same type and
/// densely packed. This is necessary for zero copy view type casting of
/// non numeric scalar types.
/// For example: Matrix<RGBASample<Float>> -> NHWCTensor<Float>
///
public protocol UniformDenseScalar: ScalarConformance, Equatable {
    associatedtype Component: AnyFixedSizeScalar
    static var componentCount: Int { get }
}

public extension UniformDenseScalar {
    static var componentCount: Int {
        return MemoryLayout<Self>.size / MemoryLayout<Component>.size
    }
}

public protocol UniformDenseScalar2: UniformDenseScalar {
    var c0: Component { get }
    var c1: Component { get }
    init(c0: Component, c1: Component)
}

public protocol UniformDenseScalar3: UniformDenseScalar {
    var c0: Component { get }
    var c1: Component { get }
    var c2: Component { get }
    init(c0: Component, c1: Component, c2: Component)
}

public protocol UniformDenseScalar4: UniformDenseScalar {
    var c0: Component { get }
    var c1: Component { get }
    var c2: Component { get }
    var c3: Component { get }
    init(c0: Component, c1: Component, c2: Component, c3: Component)
}

//==============================================================================
// RGBImageSample
public protocol RGBImageSample: UniformDenseScalar3 {
    var r: Component { get set }
    var g: Component { get set }
    var b: Component { get set }
    init(r: Component, g: Component, b: Component)
}

public extension RGBImageSample {
    var c0: Component { return r }
    var c1: Component { return g }
    var c2: Component { return b }

    @inlinable @inline(__always)
    init(c0: Component, c1: Component, c2: Component) {
        self.init(r: c0, g: c1, b: c2)
    }
}

public struct RGBSample<Component>: RGBImageSample
where Component: AnyNumeric & AnyFixedSizeScalar {
    public var r, g, b: Component

    @inlinable @inline(__always)
    public init() { r = Component.zero; g = Component.zero; b = Component.zero }

    @inlinable @inline(__always)
    public init(r: Component, g: Component, b: Component) {
        self.r = r
        self.g = g
        self.b = b
    }
}

//==============================================================================
// RGBAImageSample
public protocol RGBAImageSample: UniformDenseScalar4 {
    var r: Component { get set }
    var g: Component { get set }
    var b: Component { get set }
    var a: Component { get set }
    init(r: Component, g: Component, b: Component, a: Component)
}

public extension RGBAImageSample {
    var c0: Component { return r }
    var c1: Component { return g }
    var c2: Component { return b }
    var c3: Component { return a }

    @inlinable @inline(__always)
    init(c0: Component, c1: Component, c2: Component, c3: Component) {
        self.init(r: c0, g: c1, b: c2, a: c3)
    }
}

public struct RGBASample<Component> : RGBAImageSample
    where Component: AnyNumeric & AnyFixedSizeScalar
{
    public var r, g, b, a: Component

    @inlinable @inline(__always)
    public init() {
        r = Component.zero
        g = Component.zero
        b = Component.zero
        a = Component.zero
    }
    
    @inlinable @inline(__always)
    public init(r: Component, g: Component, b: Component, a: Component) {
        self.r = r
        self.g = g
        self.b = b
        self.a = a
    }
}

//==============================================================================
// Audio sample types
public protocol StereoAudioSample: UniformDenseScalar2 {
    var left: Component { get set }
    var right: Component { get set }
    init(left: Component, right: Component)
}

public extension StereoAudioSample {
    var c0: Component { return left }
    var c1: Component { return right }

    @inlinable @inline(__always)
    init(c0: Component, c1: Component) {
        self.init(left: c0, right: c1)
    }
}

public struct StereoSample<Component>: StereoAudioSample
where Component: AnyNumeric & AnyFixedSizeScalar {
    public var left, right: Component

    @inlinable @inline(__always)
    public init() { left = Component.zero; right = Component.zero }

    @inlinable @inline(__always)
    public init(left: Component, right: Component) {
        self.left = left
        self.right = right
    }
}

