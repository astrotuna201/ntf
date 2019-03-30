//******************************************************************************
//  Created by Edward Connell on 3/30/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// Memory sizes
extension Int {
    var KB: Int { return self * 1024 }
    var MB: Int { return self * 1024 * 1024 }
    var GB: Int { return self * 1024 * 1024 * 1024 }
    var TB: Int { return self * 1024 * 1024 * 1024 * 1024 }
}

//==============================================================================
// String(timeInterval:
extension String {
    public init(timeInterval: TimeInterval) {
        let milliseconds = Int(timeInterval.truncatingRemainder(dividingBy: 1.0) * 1000)
        let interval = Int(timeInterval)
        let seconds = interval % 60
        let minutes = (interval / 60) % 60
        let hours = (interval / 3600)
        self = String(format: "%0.2d:%0.2d:%0.2d.%0.3d",
                      hours, minutes, seconds, milliseconds)
    }
}

//==============================================================================
// almostEquals
public func almostEquals<T: AnyNumeric>(_ a: T, _ b: T,
                                       tolerance: Double = 0.00001) -> Bool {
    return abs(a.asDouble - b.asDouble) < tolerance
}
