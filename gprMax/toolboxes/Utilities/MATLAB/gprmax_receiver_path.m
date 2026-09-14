function path = gprmax_receiver_path(filename, receiverName, gridPath)
%GPRMAX_RECEIVER_PATH Resolve a unique receiver Name, independent of rxN.
%   PATH = GPRMAX_RECEIVER_PATH(FILE, NAME, GRID) searches the main grid by
%   default, or a grid such as '/subgrids/fine'. Missing/duplicate names fail.
if nargin < 3
    gridPath = "/";
end
root = "/" + strip(string(gridPath), 'both', '/') + "/rxs";
root = replace(root, "//", "/");
info = h5info(filename, char(root));
matches = strings(0, 1);
for index = 1:numel(info.Groups)
    group = info.Groups(index);
    names = string({group.Attributes.Name});
    attribute = find(names == "Name");
    if numel(attribute) == 1 && string(group.Attributes(attribute).Value) == string(receiverName)
        matches(end + 1, 1) = string(group.Name); %#ok<AGROW>
    end
end
if numel(matches) ~= 1
    error('gprMax:MATLAB:ReceiverIdentity', ...
        'Receiver Name "%s" has %d matches in %s; use a unique name or an explicit Path.', ...
        receiverName, numel(matches), root);
end
path = matches(1);
end
