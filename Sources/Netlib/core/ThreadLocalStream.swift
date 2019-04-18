//******************************************************************************
//  Created by Edward Connell on 3/25/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
//  This is inspired by the S4TF CompilerRuntime.swift
//  thread local implementation
import Foundation

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Darwin
#else
import Glibc
#endif

//==============================================================================
/// Executes a closure on the specified stream
/// - Parameter stream: the stream to set as the `currentStream`
/// - Parameter logInfo: optional logging, if nil then it is inherited from
///             the previous scope
/// - Parameter handler: a handler for asynchronous errors, if nil it is
///             inherited from the previous scope
/// - Parameter body: A closure whose operations are to be executed on the
///             specified stream
@inline(never)
public func using<R>(_ stream: DeviceStream,
                     logInfo: LogInfo? = nil,
                     handler: StreamExceptionHandler? = nil,
                     perform body: () throws -> R) rethrows -> R {
    // sets the default stream and logging info for the current scope
    _ThreadLocalStream.value.push(stream: stream, logInfo: logInfo)
    defer { _ThreadLocalStream.value.popStream() }
    // execute the body
    return try body()
}

public typealias StreamExceptionHandler = (Error) -> Void

//==============================================================================
/// handleStreamExceptions
public func handleStreamExceptions(handler: @escaping StreamExceptionHandler) {
    let index = _ThreadLocalStream.value.streamScope.count - 1
    _ThreadLocalStream.value.streamScope[index].exceptionHandler = handler
}

//==============================================================================
/// _ThreadLocalStream

/// Manages the current scope for the current stream, log, and error handlers
@usableFromInline
class _ThreadLocalStream {
    // types
    struct Scope {
        let stream: DeviceStream
        let logInfo: LogInfo
        var exceptionHandler: StreamExceptionHandler?
    }
    
    //--------------------------------------------------------------------------
    /// stack of default device streams, logging, and exception handler
    public fileprivate(set) var streamScope: [Scope] = [
        Scope(stream: Platform.defaultStream,
              logInfo: Platform.defaultStream.logInfo,
              exceptionHandler: nil)
    ]

    //--------------------------------------------------------------------------
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
    public var currentStream: DeviceStream { return streamScope.last!.stream }
    public var currentLogInfo: LogInfo { return streamScope.last!.logInfo }

    //--------------------------------------------------------------------------
    /// push(stream:
    /// pushes the specified stream onto a stream stack which makes
    /// it the current stream used by operator functions
    @usableFromInline
    func push(stream: DeviceStream, logInfo: LogInfo? = nil) {
        streamScope.append(
            Scope(stream: stream,
                  logInfo: logInfo ?? streamScope.last!.logInfo,
                  exceptionHandler: streamScope.last!.exceptionHandler))
    }
    
    //--------------------------------------------------------------------------
    /// popStream
    /// restores the previous current stream
    @usableFromInline
    func popStream() {
        assert(streamScope.count > 1)
        _ = streamScope.popLast()
    }

    //--------------------------------------------------------------------------
    /// catchError
    /// this is used inside operator implementations to catch asynchronous
    /// errors and propagate them back to the user
    @usableFromInline
    func catchError(perform body: (DeviceStream) throws -> Void) {
        do {
            try body(currentStream)
        } catch {
            // write the error to the log
            currentLogInfo.log.write(level: .error,
                                     message: String(describing: error))
            
            // call the handler if there is one
            if let handler = streamScope.last!.exceptionHandler {
                DispatchQueue.main.sync {
                    handler(error)
                }
            } else {
                // if there is no handler then break to the debugger
                raise(SIGINT)
            }
        }
    }
    
    //--------------------------------------------------------------------------
    // shared singleton initializer
    @usableFromInline
    static var value: _ThreadLocalStream {
        // try to get an existing state
        if let state = pthread_getspecific(key) {
            return Unmanaged.fromOpaque(state).takeUnretainedValue()
        } else {
            // create and return new state
            let state = _ThreadLocalStream()
            pthread_setspecific(key, Unmanaged.passRetained(state).toOpaque())
            return state
        }
    }
}
