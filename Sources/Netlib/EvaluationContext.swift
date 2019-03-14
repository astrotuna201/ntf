//******************************************************************************
//  Created by Edward Connell on 3/12/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// A value that indicates if the model will be used for training or inference
/// This effects resource allocation during `init` and control logic in
/// the `applied(to:` function
public enum EvaluationMode { case inference, training }

//==============================================================================
/// A context contains device specific resources (e.g. CPU, GPU, cloud)
/// required to generically execute layers
open class EvaluationContext: ObjectTracking, Logging {
    /// The current evaluation mode
    public let evaluationMode: EvaluationMode
    /// The current event log
    public private(set) var log: Log?
    /// The log reporting level for this object
    public var logLevel: LogLevel
    /// The log formatted nesting level for this object
    public let nestingLevel: Int = 0

    // object tracking
    public private(set) var trackingId = 0
    public var namePath: String

    /// Creates a context.
    /// - Parameter evaluationMode: The current evaluation mode
    public init(evaluationMode: EvaluationMode, name: String?,
                log: Log?, logLevel: LogLevel = .error) {
        // assign
        let rootNamePath = name ?? String(describing: EvaluationContext.self)
        self.namePath = rootNamePath
        self.evaluationMode = evaluationMode
        self.log = log ?? Log(parentNamePath: rootNamePath)
        self.logLevel = logLevel
    }

    /// Creates a context by copying all information from an existing context.
    /// - Parameter context: The existing context to copy from.
    public required init(other: EvaluationContext) {
        self.evaluationMode = other.evaluationMode
        self.log = other.log
        self.logLevel = other.logLevel
        self.namePath = other.namePath
    }
}

//==============================================================================
/// A context conforming to a Training tag to enable
/// model conformance specialization 
public protocol Training { }

public final class TrainingContext: EvaluationContext, Training {
    public init(name: String?, log: Log?, logLevel: LogLevel = .error) {
        super.init(evaluationMode: .training, name: name,
                   log: log, logLevel: logLevel)
    }
    public required init(other: EvaluationContext) {
        super.init(other: other)
    }
}

//==============================================================================
/// A context conforming to a Inferring tag to enable
/// model conformance specialization 
public protocol Inferring { }

public final class InferenceContext: EvaluationContext, Inferring {
    public init(name: String?, log: Log?, logLevel: LogLevel = .error) {
        super.init(evaluationMode: .inference, name: name,
                   log: log, logLevel: logLevel)
    }
    public required init(other: EvaluationContext) {
        super.init(other: other)
    }
}
