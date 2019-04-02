//******************************************************************************
//  Created by Edward Connell on 3/25/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
//  This is inspired by the S4TF CompilerRuntime.swift
//  thread local implementation
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Darwin
#else
import Glibc
#endif

//==============================================================================
/// Executes a closure on the specified stream
///
/// - Parameters:
///   - body: A closure whose operations are to be executed on the
///           specified stream
@inline(never)
public func using<R>(_ stream: DeviceStream, logInfo: LogInfo? = nil,
                     perform body: () throws -> R) throws -> R {
    // sets the default stream and logging info for the current scope
    _ThreadLocal.value.push(stream: stream, logInfo: logInfo)
    defer { _ThreadLocal.value.popStream() }
    // execute the body
    let result = try body()
    // if one of the non-throwing operators like `+` within the scope
    // fails, the error is recorded and is propagated
    if let lastError = _ThreadLocal.value.lastError {
        throw lastError
    }
    return result
}

//==============================================================================
/// Executes a closure on the default stream. This is only necessary if
/// catching Errors from operator expressions is desired `+`
///
/// - Parameters:
///   - body: A closure whose operations are to be executed on the
///           default stream
@inline(never)
public func usingDefaultStream<R>(
    logInfo: LogInfo? = nil,
    perform body: () throws -> R) throws -> R {
    
    // execute the body
    let result = try body()
    // if one of the non-throwing operators like `+` within the scope
    // fails, the error is recorded and is propagated
    if let lastError = _ThreadLocal.value.lastError {
        throw lastError
    }
    return result
}

//==============================================================================
// _ThreadLocal
@usableFromInline
class _ThreadLocal {
    //--------------------------------------------------------------------------
    // properties
    /// stack of default device streams and logging
    var streamScope: [(stream: DeviceStream, logInfo: LogInfo, error: Error?)] =
        [(Platform.defaultStream, Platform.defaultStream.logging!, nil)]

    /// thread data key
    private static let key: pthread_key_t = {
        var key = pthread_key_t()
        pthread_key_create(&key) {
            #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
            let _: AnyObject = Unmanaged.fromOpaque($0).takeRetainedValue()
            #else
            let _: AnyObject = Unmanaged.fromOpaque($0!).takeRetainedValue()
            #endif
        }
        return key
    }()
    
    // there will always be the platform default stream and logInfo
    public var defaultStream: DeviceStream { return streamScope.last!.stream }
    public var defaultLogInfo: LogInfo { return streamScope.last!.logInfo }
    public var lastError: Error? { return streamScope.last!.error }
    public var noError: Bool { return streamScope.last!.error == nil }
    public var errorOccurred: Bool { return streamScope.last!.error != nil }

    //--------------------------------------------------------------------------
    // stack functions
    @usableFromInline
    func push(stream: DeviceStream, logInfo: LogInfo? = nil) {
        let info = logInfo ?? streamScope.last!.logInfo
        streamScope.append((stream, info, nil))
    }
    
    @usableFromInline
    func popStream() {
        assert(streamScope.count > 1)
        _ = streamScope.popLast()
    }

    /// this helper is used for static operators that perform throwing
    /// operations but cannot throw themselves due to protocol conformance
    /// like AdditiveArithmetic. If an exception is thrown, this catches and
    /// stores the first Error and subsequent body calls are skipped. The
    /// error is then thrown by the using(stream function.
    public func catchError<T: TensorView>(
        perform body: () throws -> T) -> T {
        // if there is an outstanding error than just return
        guard lastError == nil else { return T() }
        // try the body
        do {
            return try body()
        } catch {
            // record the error
            streamScope[(streamScope.count - 1)].error = error
            return T()
        }
    }

    //--------------------------------------------------------------------------
    // shared singleton initializer
    @usableFromInline
    static var value: _ThreadLocal {
        // try to get an existing state
        if let state = pthread_getspecific(key) {
            return Unmanaged.fromOpaque(state).takeUnretainedValue()
        } else {
            // create and return new state
            let state = _ThreadLocal()
            pthread_setspecific(key, Unmanaged.passRetained(state).toOpaque())
            return state
        }
    }
}
