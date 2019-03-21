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
public protocol Evaluating {}

open class EvaluationContext: Evaluating, ObjectTracking, Logging {
    /// The current evaluation mode
    public let evaluationMode: EvaluationMode
    /// The current event log
    public private(set) var log: Log?
    /// The log reporting level for this object
    public var logLevel: LogLevel
    /// The log formatted nesting level for this object
    public let nestingLevel: Int = 0
    /// The platform instance
    public private(set) var platform: Platform! = nil

    // object tracking
    public private(set) var trackingId = 0
    public var namePath: String

    /// Creates a context.
    /// - Parameter evaluationMode: The current evaluation mode
    public init(evaluationMode: EvaluationMode, name: String,
                log: Log?, logLevel: LogLevel = .error) {
        // assign
        self.namePath = name
        self.evaluationMode = evaluationMode
        self.log = log ?? Log(parentNamePath: name)
        self.logLevel = logLevel
        self.platform = Platform(context: self)
    }

    /// Creates a context by copying all information from an existing context.
    /// - Parameter context: The existing context to copy from.
    public required init(other: EvaluationContext) {
        self.evaluationMode = other.evaluationMode
        self.log = other.log
        self.logLevel = other.logLevel
        self.namePath = other.namePath
        self.platform = other.platform
    }
}

//==============================================================================
/// A context conforming to a Training tag to enable
/// model conformance specialization 
public protocol Training: Evaluating { }

public final class TrainingContext: EvaluationContext, Training {
    public init(name: String? = nil, log: Log? = nil, logLevel: LogLevel = .error) {
        super.init(evaluationMode: .training,
                   name: name ?? String(describing: TrainingContext.self),
                   log: log, logLevel: logLevel)
    }
    public required init(other: EvaluationContext) {
        super.init(other: other)
    }
}

//==============================================================================
/// A context conforming to a Inferring tag to enable
/// model conformance specialization 
public protocol Inferring: Evaluating { }

public final class InferenceContext: EvaluationContext, Inferring {
    public init(name: String? = nil, log: Log? = nil, logLevel: LogLevel = .error) {
        super.init(evaluationMode: .inference,
                   name: name ?? String(describing: InferenceContext.self),
                   log: log, logLevel: logLevel)
    }
    public required init(other: EvaluationContext) {
        super.init(other: other)
    }
}
