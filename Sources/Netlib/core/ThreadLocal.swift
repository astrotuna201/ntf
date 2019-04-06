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
    return try body()
}

//==============================================================================
// _ThreadLocal
@usableFromInline
class _ThreadLocal {
    //--------------------------------------------------------------------------
    // properties
    /// stack of default device streams and logging
    var streamScope: [(stream: DeviceStream, logInfo: LogInfo)] =
        [(Platform.defaultStream, Platform.defaultStream.logging!)]

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
    public var defaultLogging: LogInfo { return streamScope.last!.logInfo }

    //--------------------------------------------------------------------------
    // stack functions
    @usableFromInline
    func push(stream: DeviceStream, logInfo: LogInfo? = nil) {
        let info = logInfo ?? streamScope.last!.logInfo
        streamScope.append((stream, info))
    }
    
    @usableFromInline
    func popStream() {
        assert(streamScope.count > 1)
        _ = streamScope.popLast()
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
