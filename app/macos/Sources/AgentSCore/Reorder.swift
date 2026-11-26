extension Array {
    /// The array with the first element matching `item` moved to the place of the first matching
    /// `target`, the elements between them sliding over: dragged down it lands after the target,
    /// dragged up it lands before. Nil when either is missing or they're the same element.
    public func moving(_ item: (Element) -> Bool, to target: (Element) -> Bool) -> [Element]? {
        guard let from = firstIndex(where: item), let to = firstIndex(where: target), from != to else { return nil }
        var result = self
        result.insert(result.remove(at: from), at: to)
        return result
    }
}
