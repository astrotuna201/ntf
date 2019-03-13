//******************************************************************************
//  Created by Edward Connell on 1/10/17
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Dispatch
import Foundation

#if os(Linux)
import Glibc
#endif

public protocol ObjectTracking : class {
    var namePath: String { get }
	var trackingId: Int { get }
}

// singleton
public var objectTracker = ObjectTracker()

//==============================================================================
/// ObjectTracker
/// The object tracker class is used during debug to track the lifetime of
/// class objects to prevent reference cycles and memory leaks
final public class ObjectTracker {
	//--------------------------------------------------------------------------
	// types
	public struct ItemInfo {
        weak var object: ObjectTracking?
		let typeName: String
		let supplementalInfo: String
		var isStatic: Bool
	}

	//--------------------------------------------------------------------------
	// properties TODO: document all these
	/// global shared instance
	public static let global = ObjectTracker()
	
	private let queue = DispatchQueue(label: "ObjectTracker.queue")
	public let counter = AtomicCounter()
	public var debuggerRegisterBreakId = -1
	public var debuggerRemoveBreakId = -1
	public private(set) var activeObjects = [Int : ItemInfo]()
	public var hasActiveObjects: Bool { return !activeObjects.isEmpty }

	//--------------------------------------------------------------------------
	// activeObjectsInfo
	public func getActiveObjectInfo(includeStatics: Bool = false) -> String {
		var result = "\n"
		var activeCount = 0
		for objectId in (activeObjects.keys.sorted { $0 < $1 }) {
			let info = activeObjects[objectId]!
			if includeStatics || !info.isStatic {
				result += getObjectDescription(id: objectId, info: info) + "\n"
				activeCount += 1
			}
		}
		if activeCount > 0 {
			result += "\nObjectTracker contains \(activeCount) live objects\n"
		}
		return result
	}

	// getObjectDescription
	private func getObjectDescription(id: Int, info: ItemInfo) -> String {
		var description = "[\(info.typeName)(\(id))"
		if info.supplementalInfo.isEmpty {
			description += "]"
		} else {
			description += " \(info.supplementalInfo)]"
		}

		if let propObject = info.object {
			description += " path: \(propObject.namePath)"
		}
		return description
	}

	//--------------------------------------------------------------------------
	// register(object:
	public func register(object: ObjectTracking, info: String = "") -> Int {
		#if ENABLE_TRACKING
            let id = counter.increment()
            register(id: id, info:
                ItemInfo(object: object, typeName: object.typeName,
                         supplementalInfo: info, isStatic: false))
            return id
        #else
            return 0
        #endif
	}

	//--------------------------------------------------------------------------
	// register(type:
	public func register<T>(type: T, info: String = "") -> Int {
		#if ENABLE_TRACKING
            let id = counter.increment()
            register(id: id, info:
                ItemInfo(object: nil,
                         typeName: String(describing: Swift.type(of: T.self)),
                         supplementalInfo: info, isStatic: false))
            return id
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
	
	//--------------------------------------------------------------------------
	// markStatic
	public func markStatic(trackingId: Int) {
		#if ENABLE_TRACKING
            _ = queue.sync { activeObjects[trackingId]!.isStatic = true }
		#endif
	}
	
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
