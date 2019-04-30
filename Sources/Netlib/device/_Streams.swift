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
    /// appThreadStream
    /// a unified memory stream used to synchronize access for the
    /// application thread
    public let appThreadStream: DeviceStream
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
    
    //--------------------------------------------------------------------------
    /// logInfo
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
        // create the default stream based on service and device priority.
        let stream = Platform.local.defaultDevice.createStream(name: "default")
        streamScope = [stream]
        
        // _Streams is a static object, so mark the default stream as static
        // so it won't show up in leak reports
        ObjectTracker.global.markStatic(trackingId: stream.trackingId)

        // use the default stream as the app stream if it uses
        // unified memory addressing
        if stream.device.memoryAddressing == .unified {
            appThreadStream = stream
            
        } else {
            // create a uma stream for app thread synchronization
            appThreadStream =
                Platform.local.createStream(deviceId: 0,
                                            serviceName: "cpu",
                                            name: "applicationThreadStream")
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
