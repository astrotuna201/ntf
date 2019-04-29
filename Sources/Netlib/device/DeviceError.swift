//******************************************************************************
//  Created by Edward Connell on 4/22/19
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// DeviceError
public enum DeviceError : Error {
    case streamError(idPath: [Int], message: String)
    case timeout(idPath: [Int], message: String)
}

public typealias DeviceErrorHandler = (Error) -> Void

public protocol DeviceErrorHandling: class, _Logging {
    /// user defined handler to override the default
    var deviceErrorHandler: DeviceErrorHandler? { get set }
    /// safe access mutex
    var _errorMutex: Mutex { get }
    /// last error recorded
    var _lastError: Error? { get set }
    
    /// handler that will either call a user handler if defined or propagate
    /// up the device tree
    func handleDevice(error: Error)
}

public extension DeviceErrorHandling {
    /// safe access
    var lastError: Error? {
        get { return _errorMutex.sync { _lastError } }
        set { _errorMutex.sync { _lastError = newValue } }
    }
    
    //--------------------------------------------------------------------------
    /// reportDevice(error:event:
    /// sets and propagates a stream error
    /// - Parameter error: the error to report
    /// - Parameter event: an optional event to signal, used to
    ///   unblock waiting threads of a stream fails
    func reportDevice(error: Error, event: StreamEvent? = nil) {
        // set the error state
        lastError = error
        
        // write the error to the log
        writeLog(String(describing: error))
        
        // propagate on app thread
        DispatchQueue.main.async {
            self.handleDevice(error: error)
        }
        
        // signal the completion event in case the app thread is waiting
        event?.signal()
    }
}

