//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//

//==============================================================================
// CpuStreamEvent
final public class CpuStreamEvent : StreamEvent {
    public required init(options: StreamEventOptions) {
        trackingId = ObjectTracker.global
            .register(self, namePath: logging?.namePath)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
    
    //--------------------------------------------------------------------------
    // properties
    public private (set) var trackingId = 0
    public var occurred: Bool { return true }
    public var logging: LogInfo? = nil
}
