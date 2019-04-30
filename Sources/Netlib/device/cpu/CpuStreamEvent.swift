//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// CpuStreamEvent
/// a stream event behaves like a barrier. The first caller to wait takes
/// the wait semaphore
final public class CpuStreamEvent : StreamEvent {
    // properties
    public private (set) var trackingId = 0
    private let accessMutex = Mutex()
    private let semaphore = DispatchSemaphore(value: 0)
    public  let stream: DeviceStream
    private var _occurred: Bool = true
    private static let name = String(describing: CpuStreamEvent.self)

    //--------------------------------------------------------------------------
    // initializers
    public init(stream: DeviceStream,
                options: StreamEventOptions = StreamEventOptions()) {
        self.stream = stream
        #if DEBUG
        trackingId = ObjectTracker.global.register(self)
        #endif
    }
    deinit {
        #if DEBUG
        ObjectTracker.global.remove(trackingId: trackingId)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // functions
    /// These are not exposed through the protocol because they should only
    /// be manipulated via the DeviceStream protocol
    public func signal() {
        semaphore.signal()
    }
    
    public var occurred: Bool {
        get { return accessMutex.sync { _occurred } }
        set { accessMutex.sync { _occurred = newValue } }
    }
    
    public func blockingWait(for timeout: TimeInterval) throws {
        try accessMutex.sync {
            guard !_occurred else { return }
            #if DEBUG
            stream.diagnostic(
                "\(CpuStreamEvent.name)(\(trackingId)) waiting...",
                categories: .streamSync)
            #endif
            
            if timeout > 0 {
                let waitUntil = DispatchWallTime.now() + (timeout * 1000000)
                if semaphore.wait(wallTimeout: waitUntil) == .timedOut {
                    #if DEBUG
                    stream.diagnostic(
                        "\(CpuStreamEvent.name)(\(trackingId)) timed out",
                        categories: .streamSync)
                    #endif
                    throw StreamEventError.timedOut
                }
            } else {
                semaphore.wait()
            }
            
            #if DEBUG
            stream.diagnostic("\(CpuStreamEvent.name)(\(trackingId)) occured",
                categories: .streamSync)
            #endif
            _occurred = true
        }
    }
}
