--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'convnet'

return function(args)
    args.n_units        = {32, 64, 64, 128}
    args.filter_size    = {8, 4, 3, 2}
    args.filter_stride  = {4, 2, 1, 1}
    args.n_hid          = {512}
    args.nl             = nn.Rectifier

    return create_network(args)
end

