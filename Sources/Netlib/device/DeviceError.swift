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
    var _deviceErrorHandler: DeviceErrorHandler! { get set }
    var _lastError: Error? { get set }
    var errorMutex: Mutex { get }
}

public extension DeviceErrorHandling {
    /// use access get/set to prevent setting `nil`
    var deviceErrorHandler: DeviceErrorHandler {
        get { return _deviceErrorHandler }
        set { _deviceErrorHandler = newValue }
    }
    
    /// safe access
    var lastError: Error? {
        get { return errorMutex.sync { _lastError } }
        set { errorMutex.sync { _lastError = newValue } }
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
            self.deviceErrorHandler(error)
        }
        
        // signal the completion event in case the app thread is waiting
        event?.signal()
    }
}

