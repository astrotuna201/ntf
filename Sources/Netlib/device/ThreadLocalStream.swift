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
/// - Parameter body: A closure whose operations are to be executed on the
///             specified stream
@inline(never)
public func using<R>(_ stream: DeviceStream,
                     perform body: () throws -> R) rethrows -> R {
    // sets the default stream and logging info for the current scope
    _Streams.local.push(stream: stream)
    defer { _Streams.local.popStream() }
    // execute the body
    return try body()
}

//==============================================================================
/// _Streams
/// Manages the scope for the current stream, log, and error handlers
@usableFromInline
class _Streams {
    /// stack of default device streams, logging, and exception handler
    var streamScope: [DeviceStream] = []

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

    //--------------------------------------------------------------------------
    /// returns the thread local instance of the streams stack
    @usableFromInline
    static var local: _Streams {
        // try to get an existing state
        if let state = pthread_getspecific(key) {
            return Unmanaged.fromOpaque(state).takeUnretainedValue()
        } else {
            // create and return new state
            let state = _Streams()
            pthread_setspecific(key, Unmanaged.passRetained(state).toOpaque())
            return state
        }
    }

    //--------------------------------------------------------------------------
    /// current
    public static var current: DeviceStream {
        return _Streams.local.streamScope.last!
    }
    
    // there will always be the platform default stream and logInfo
    public var logInfo: LogInfo { return streamScope.last!.logInfo }

    //--------------------------------------------------------------------------
    /// updateDefault
    public func updateDefault(stream: DeviceStream) {
        streamScope[0] = stream
    }
    
    //--------------------------------------------------------------------------
    // initializers
    private init() {
        do {
            let stream = try
                Platform.local.defaultDevice.createStream(name: "default")
            streamScope = [stream]
        } catch {
            Platform.local.reportDevice(error: error)
        }
    }
    
    //--------------------------------------------------------------------------
    /// push(stream:
    /// pushes the specified stream onto a stream stack which makes
    /// it the current stream used by operator functions
    @usableFromInline
    func push(stream: DeviceStream) {
        streamScope.append(stream)
    }
    
    //--------------------------------------------------------------------------
    /// popStream
    /// restores the previous current stream
    @usableFromInline
    func popStream() {
        assert(streamScope.count > 1)
        _ = streamScope.popLast()
    }
}
