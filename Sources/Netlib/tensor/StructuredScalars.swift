//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// conformance indicates that scalar components are of the same type and
/// densely packed. This is necessary for zero copy view type casting of
/// short vector Element types.
/// For example: Matrix<RGBA<Float>> -> NHWCTensor<Float>
///
public protocol UniformDenseScalar: Equatable {
    associatedtype Component
    static var componentCount: Int { get }
}

public extension UniformDenseScalar {
    static var componentCount: Int {
        return MemoryLayout<Self>.size / MemoryLayout<Component>.size
    }
}

//==============================================================================
// RGB
public protocol RGBProtocol: UniformDenseScalar {}

public struct RGB<Component>: RGBProtocol where Component: Numeric {
    public var r, g, b: Component

    @inlinable @inline(__always)
    public init() { r = Component.zero; g = Component.zero; b = Component.zero }

    @inlinable @inline(__always)
    public init(r: Component, g: Component, b: Component) {
        self.r = r; self.g = g; self.b = b
    }
}

//==============================================================================
// RGBA
public protocol RGBAProtocol: UniformDenseScalar {}

public struct RGBA<Component> : RGBAProtocol where Component: Numeric {
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
        self.r = r; self.g = g; self.b = b; self.a = a
    }
}

//==============================================================================
// Stereo
public protocol StereoProtocol: UniformDenseScalar {}

public struct Stereo<Component>: StereoProtocol where Component: Numeric {
    public var left, right: Component

    @inlinable @inline(__always)
    public init() { left = Component.zero; right = Component.zero }

    @inlinable @inline(__always)
    public init(left: Component, right: Component) {
        self.left = left; self.right = right
    }
}

