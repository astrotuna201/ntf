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
public protocol UniformDenseScalar: ScalarConformance {
    associatedtype Component: AnyFixedSizeScalar
    static var componentCount: Int { get }
}

public extension UniformDenseScalar {
    static var componentCount: Int {
        return MemoryLayout<Self>.size / MemoryLayout<Component>.size
    }
}

public protocol UniformDenseScalar2: UniformDenseScalar {
    var c0: Component { get set }
    var c1: Component { get set }
}

public protocol UniformDenseScalar3: UniformDenseScalar {
    var c0: Component { get set }
    var c1: Component { get set }
    var c2: Component { get set }
}

public protocol UniformDenseScalar4: UniformDenseScalar {
    var c0: Component { get set }
    var c1: Component { get set }
    var c2: Component { get set }
    var c3: Component { get set }
}

//==============================================================================
// RGBImageSample
public protocol RGBImageSample: UniformDenseScalar {
    var r: Component { get set }
    var g: Component { get set }
    var b: Component { get set }
}

public struct RGBSample<Component>: RGBImageSample
where Component: AnyNumeric & AnyFixedSizeScalar {
    public var r, g, b: Component
    public init() { r = Component.zero; g = Component.zero; b = Component.zero }
    
    public init(r: Component, g: Component, b: Component) {
        self.r = r
        self.g = g
        self.b = b
    }
}

public protocol RGBAImageSample: UniformDenseScalar {
    var r: Component { get set }
    var g: Component { get set }
    var b: Component { get set }
    var a: Component { get set }
}

public struct RGBASample<Component> : RGBAImageSample
where Component: AnyNumeric & AnyFixedSizeScalar {
    public var r, g, b, a: Component
    public init() {
        r = Component.zero
        g = Component.zero
        b = Component.zero
        a = Component.zero
    }
    
    public init(r: Component, g: Component, b: Component, a: Component) {
        self.r = r
        self.g = g
        self.b = b
        self.a = a
    }
}

//==============================================================================
// Audio sample types
public protocol StereoAudioSample: UniformDenseScalar {
    var left: Component { get set }
    var right: Component { get set }
}

public struct StereoSample<Component>: StereoAudioSample
where Component: AnyNumeric & AnyFixedSizeScalar {
    public var left, right: Component
    public init() { left = Component.zero; right = Component.zero }
}

