//******************************************************************************
//  Created by Edward Connell on  3/10/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation
import TensorFlow

public protocol Function :
    Differentiable,
    KeyPathIterable where Self.AllDifferentiableVariables : KeyPathIterable {

    //--------------------------------------------------------------------------
    // types
    /// The input type of the function.
    associatedtype Input : Differentiable
    /// The output type of the function.
    associatedtype Output : Differentiable

    //--------------------------------------------------------------------------
    // properties
    /// the unique identifier for this function
    var id: UUID { get }

    /// This is the reusable output object allocated by the function
    /// during `setup`. This avoids allocating a new output for every iteration
    /// of the applied function
    var output: Self.Output { get }
    
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
