//******************************************************************************
//  Created by Edward Connell on 1/10/17
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Dispatch
import Foundation

#if os(Linux)
import Glibc
#endif

public protocol ObjectTracking: class {
	var trackingId: Int { get }
}

//==============================================================================
/// ObjectTracker
/// The object tracker class is used during debug to track the lifetime of
/// class objects to prevent reference cycles and memory leaks
final public class ObjectTracker {
	//--------------------------------------------------------------------------
	// types
	public struct ItemInfo {
        var isStatic: Bool
        let namePath: String?
		let supplementalInfo: String?
        let typeName: String
	}

    // singleton
    private init() {}
    
	//--------------------------------------------------------------------------
	// properties
	/// global shared instance
	public static let global = ObjectTracker()
    /// thread access queue
	private let queue = DispatchQueue(label: "ObjectTracker.queue")
    /// counter to generate object ids
	public let counter = AtomicCounter()
    /// init registration tracking id to break on
	public var debuggerRegisterBreakId = -1
    /// deinit break id
	public var debuggerRemoveBreakId = -1
    // the list of currently registered objects
	public private(set) var activeObjects = [Int: ItemInfo]()
    /// true if there are currently unreleased objects
	public var hasActiveObjects: Bool { return !activeObjects.isEmpty }

	//--------------------------------------------------------------------------
	// activeObjectsInfo
	public func getActiveObjectInfo(includeStatics: Bool = false) -> String {
		var result = "\n"
		var activeCount = 0
		for objectId in (activeObjects.keys.sorted { $0 < $1 }) {
			let info = activeObjects[objectId]!
			if includeStatics || !info.isStatic {
				result += getObjectDescription(trackingId: objectId,
                                               info: info) + "\n"
				activeCount += 1
			}
		}
		if activeCount > 0 {
			result += "\nObjectTracker contains \(activeCount) live objects\n"
		}
		return result
	}

	// getObjectDescription
	private func getObjectDescription(trackingId: Int, info: ItemInfo) -> String {
        return "[\(info.typeName)(\(trackingId))" +
            (info.supplementalInfo == nil ? "]":" \(info.supplementalInfo!)]") +
            (info.namePath == nil ? "" : " path: \(info.namePath!)")
	}

	//--------------------------------------------------------------------------
	// register(object:
    public func register(_ object: ObjectTracking,
                         namePath: String? = nil,
                         supplementalInfo: String? = nil,
                         isStatic: Bool = false) -> Int {
        #if ENABLE_TRACKING
        let info = ItemInfo(isStatic: isStatic,
                            namePath: namePath,
                            supplementalInfo: supplementalInfo,
                            typeName: String(describing: object.self))
        let trackingId = counter.increment()

        register(trackingId: trackingId, info: info)
        return trackingId
        
        #else
        return 0
        #endif
    }

    //--------------------------------------------------------------------------
    // register
    private func register(trackingId: Int, info: ItemInfo) {
        queue.sync {
            if trackingId == debuggerRegisterBreakId {
                print("ObjectTracker debug break for id(\(trackingId))")
                raise(SIGINT)
            }
            activeObjects[trackingId] = info
        }
    }
    
    // TODO: maybe remove this if unused? If so, combine register functions above
//    //--------------------------------------------------------------------------
//    // register(type:
//    public func register<T>(type: T, info: String = "") -> Int {
//        #if ENABLE_TRACKING
//            let trackingId = counter.increment()
//            register(trackingId: trackingId, info:
//                ItemInfo(object: nil,
//                         typeName: String(describing: Swift.type(of: T.self)),
//                         supplementalInfo: info, isStatic: false))
//            return id
//        #else
//            return 0
//        #endif
//    }

    // TODO: maybe remove this if unused?
//    //--------------------------------------------------------------------------
//    // markStatic
//    public func markStatic(trackingId: Int) {
//        #if ENABLE_TRACKING
//            _ = queue.sync { activeObjects[trackingId]!.isStatic = true }
//        #endif
//    }

	//--------------------------------------------------------------------------
	// remove
	public func remove(trackingId: Int) {
		#if ENABLE_TRACKING
        _ = queue.sync {
            if trackingId == debuggerRemoveBreakId {
                print("ObjectTracker debug break remove for id(\(trackingId))")
                raise(SIGINT)
            }
            activeObjects.removeValue(forKey: trackingId)
        }
		#endif
	}
}
