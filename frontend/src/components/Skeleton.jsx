const Skeleton = ({ width = '100%', height = '16px', radius }) => (
  <div
    className="skeleton"
    style={{ width, height, borderRadius: radius || undefined }}
  />
);

export default Skeleton;
