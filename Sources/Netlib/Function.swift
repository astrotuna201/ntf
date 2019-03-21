//******************************************************************************
//  Created by Edward Connell on  3/10/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation
import TensorFlow

/// A value that indicates if the model will be used for training or inference
/// This effects resource allocation during `init` and control logic in
/// the `applied(to:` function
public enum EvaluationMode { case inference, training }

/// Function
public protocol Function: Differentiable,
    KeyPathIterable where Self.AllDifferentiableVariables: KeyPathIterable {

    //--------------------------------------------------------------------------
    // types
    /// The input type of the function.
    associatedtype Input: Differentiable
    /// The output type of the function.
    associatedtype Output: Differentiable

    //--------------------------------------------------------------------------
    // properties
    /// This is the reusable output object allocated by the function
    /// during `setup`. This avoids allocating a new output for every iteration
    /// of the applied function
    var output: Self.Output { get }
    /// The device stream used to execute the function isntance
    var stream: DeviceStream { get }

    /// Returns the output obtained from applying the function to the given input.
    ///
    /// - Parameters
    ///   - input: The input to the function.
    ///   - context: Device specific configuration and resources for
    ///              function execution.
    /// - Returns: The output.
    @differentiable
    func applied(to input: Self.Input) -> Self.Output
}

