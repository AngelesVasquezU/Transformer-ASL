auto h = x_pos;
for (auto &blk : blocks)
{
    h = blk->forward(h);
}
